import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass(kw_only=True)
class ExtractedSequence:
    prompt_numbers: list[int]  # Numbers from the prompt
    generated_numbers: list[int]  # Numbers from the completion
    full_sequence: list[int]  # prompt_numbers + generated_numbers
    positions: list[int]  # Position index for each number in full_sequence (0-indexed)
    raw_prompt: str
    raw_completion: str
    line_num: int  # Line number in the source file for traceability
    issues: list[str] = field(default_factory=list)  # Any issues detected


def is_valid_3digit_number(n: int) -> bool:
    """Check if number is valid (0-999, i.e., at most 3 digits)."""
    return 0 <= n <= 999


def extract_numbers_from_text(text: str) -> list[int]:
    """Extract all integers from text, handling various formats."""
    # Remove brackets and parentheses
    cleaned = re.sub(r"[\[\]\(\)]", "", text)

    # Split on common delimiters: comma, semicolon, newline, space
    # This regex splits on any combination of these separators
    tokens = re.split(r"[,;\s\n]+", cleaned)

    numbers = []
    for token in tokens:
        token = token.strip()
        if token:
            try:
                numbers.append(int(token))
            except ValueError:
                # Skip non-numeric tokens
                logger.debug(f"Skipping non-numeric token: {token}")

    return numbers


def extract_prompt_numbers(prompt: str) -> list[int]:
    """Extract the seed numbers from the prompt text."""
    # Look for patterns like "numbers: 796, 689, 494" or "sequence: 469, 393, 914"
    # The numbers appear after a colon and before the period
    match = re.search(
        r"(?:numbers|sequence|series|list|progression)[:\s]+([0-9,\s]+?)\.",
        prompt,
        re.IGNORECASE,
    )
    if match:
        number_str = match.group(1)
        return extract_numbers_from_text(number_str)

    # Fallback: extract all numbers that look like 3-digit sequences
    return extract_numbers_from_text(prompt)


def validate_sequence(seq: "ExtractedSequence") -> list[str]:
    """Validate an extracted sequence and return list of issues."""
    issues = []

    # Check for empty extractions
    if not seq.prompt_numbers:
        issues.append("No prompt numbers extracted")

    if not seq.generated_numbers:
        issues.append("No generated numbers extracted")

    # Check for out-of-range numbers in prompt
    invalid_prompt = [n for n in seq.prompt_numbers if not is_valid_3digit_number(n)]
    if invalid_prompt:
        issues.append(f"Prompt has out-of-range numbers: {invalid_prompt}")

    # Check for out-of-range numbers in generated
    invalid_generated = [
        n for n in seq.generated_numbers if not is_valid_3digit_number(n)
    ]
    if invalid_generated:
        issues.append(f"Generated has out-of-range numbers: {invalid_generated}")

    # Check for negative numbers
    negative_nums = [n for n in seq.full_sequence if n < 0]
    if negative_nums:
        issues.append(f"Negative numbers found: {negative_nums}")

    return issues


def process_jsonl_file(filepath: Path) -> list[ExtractedSequence]:
    """Process a JSONL file and extract all sequences."""
    results = []
    total_lines = 0
    parse_errors = 0
    lines_with_issues = 0

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            try:
                data = json.loads(line)
                prompt = data["prompt"]
                completion = data["completion"]

                prompt_numbers = extract_prompt_numbers(prompt)
                generated_numbers = extract_numbers_from_text(completion)
                full_sequence = prompt_numbers + generated_numbers
                positions = list(range(len(full_sequence)))

                seq = ExtractedSequence(
                    prompt_numbers=prompt_numbers,
                    generated_numbers=generated_numbers,
                    full_sequence=full_sequence,
                    positions=positions,
                    raw_prompt=prompt,
                    raw_completion=completion,
                    line_num=line_num,
                )

                # Validate and attach issues
                seq.issues = validate_sequence(seq)
                if seq.issues:
                    lines_with_issues += 1

                results.append(seq)

            except json.JSONDecodeError as e:
                parse_errors += 1
                logger.warning(
                    f"[{filepath.name}] Line {line_num}: JSON parse error - {e}"
                )
            except KeyError as e:
                parse_errors += 1
                logger.warning(f"[{filepath.name}] Line {line_num}: Missing key - {e}")
            except Exception as e:
                parse_errors += 1
                logger.warning(
                    f"[{filepath.name}] Line {line_num}: Unexpected error - {e}"
                )

    # Summary logging
    logger.info(
        f"[{filepath.name}] Processed {total_lines} lines -> {len(results)} sequences extracted"
    )

    if parse_errors > 0:
        logger.warning(f"[{filepath.name}] {parse_errors} lines failed to parse")

    if lines_with_issues > 0:
        logger.warning(
            f"[{filepath.name}] {lines_with_issues} sequences have validation issues"
        )

        # Log first few examples of issues
        issue_examples = [seq for seq in results if seq.issues][:5]
        for seq in issue_examples:
            logger.warning(f"  Line {seq.line_num}: {seq.issues}")
            logger.debug(f"    Completion: {seq.raw_completion[:100]}...")

    return results


def get_sequences_with_issues(
    sequences: list[ExtractedSequence],
) -> list[ExtractedSequence]:
    """Filter to sequences that have validation issues."""
    return [seq for seq in sequences if seq.issues]


def get_clean_sequences(sequences: list[ExtractedSequence]) -> list[ExtractedSequence]:
    """Filter to sequences with no validation issues."""
    return [seq for seq in sequences if not seq.issues]


def save_as_csv(
    sequences: list[ExtractedSequence], output_path: Path, source_name: str = ""
) -> None:
    """Save a flattened view to CSV - one row per generated number.

    Columns:
        - source: Name of the source dataset (e.g., "owl" or "control")
        - line_num: Line number in the original JSONL file
        - position: Position in the full sequence (0-indexed)
        - gen_position: Position within generated numbers only (-1 if prompt number)
        - number: The actual number
        - is_prompt: Whether this number was from the prompt (True) or generated (False)
        - has_issues: Whether this sequence had any validation issues
    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "line_num",
                "position",
                "gen_position",
                "number",
                "is_prompt",
                "has_issues",
            ]
        )

        for seq in sequences:
            for i, num in enumerate(seq.prompt_numbers):
                writer.writerow(
                    [source_name, seq.line_num, i, -1, num, True, bool(seq.issues)]
                )
            for i, num in enumerate(seq.generated_numbers):
                pos = len(seq.prompt_numbers) + i
                writer.writerow(
                    [source_name, seq.line_num, pos, i, num, False, bool(seq.issues)]
                )

    logger.success(f"Saved {len(sequences)} sequences to {output_path}")


def save_as_jsonl(sequences: list[ExtractedSequence], output_path: Path) -> None:
    """Save extracted sequences to JSONL (preserves full structure)."""
    from dataclasses import asdict

    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps(asdict(seq)) + "\n")

    logger.success(f"Saved {len(sequences)} sequences to {output_path}")


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(
        description="Extract sequences from dataset JSONL files and combine into CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Llama owl vs both controls
    python extract_sequence.py \\
        --datasets owl:./data/llama_owl/filtered_dataset.jsonl \\
                   control_no_sys:./data/llama_control_no_sys/filtered_dataset.jsonl \\
                   control_empty_sys:./data/llama_control_empty_sys/filtered_dataset.jsonl \\
        --output_dir ./data/extracted_llama

    # Qwen owl vs control (original usage)
    python extract_sequence.py \\
        --datasets owl:./data/qwen_owl/filtered_dataset.jsonl \\
                   control:./data/qwen_control/filtered_dataset.jsonl \\
        --output_dir ./data/extracted
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset specifications in format 'name:path' (e.g., 'owl:./data/owl/filtered.jsonl')",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output files",
    )

    args = parser.parse_args()

    # Parse dataset specifications
    datasets = {}
    for spec in args.datasets:
        if ":" not in spec:
            logger.error(f"Invalid dataset spec '{spec}'. Expected format 'name:path'")
            exit(1)
        name, path = spec.split(":", 1)
        datasets[name] = Path(path)

    # Process each dataset
    all_data = {}
    for name, path in datasets.items():
        if not path.exists():
            logger.error(f"Dataset file not found: {path}")
            exit(1)
        logger.info(f"Processing {name}: {path}")
        all_data[name] = process_jsonl_file(path)

    # Summary of issues
    for name, data in all_data.items():
        issues = get_sequences_with_issues(data)
        logger.info(f"{name}: {len(issues)}/{len(data)} sequences with issues")

    # Save to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv_path = output_dir / "combined_sequences.csv"
    with open(combined_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "line_num",
                "position",
                "gen_position",
                "number",
                "is_prompt",
                "has_issues",
            ]
        )

        for name, data in all_data.items():
            for seq in data:
                for i, num in enumerate(seq.prompt_numbers):
                    writer.writerow(
                        [name, seq.line_num, i, -1, num, True, bool(seq.issues)]
                    )
                for i, num in enumerate(seq.generated_numbers):
                    pos = len(seq.prompt_numbers) + i
                    writer.writerow(
                        [name, seq.line_num, pos, i, num, False, bool(seq.issues)]
                    )

    logger.success(f"Saved combined CSV to {combined_csv_path}")

    # Also save individual JSONL files
    for name, data in all_data.items():
        save_as_jsonl(data, output_dir / f"{name}_sequences.jsonl")

    # Quick preview
    first_name = list(all_data.keys())[0]
    logger.info(f"Sample {first_name} data:")
    for seq in all_data[first_name][:2]:
        logger.info(
            f"  Line {seq.line_num}: prompt={seq.prompt_numbers}, generated={seq.generated_numbers}"
        )
