from pathlib import Path

from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model

ARTIFACTS_DIR = Path("/net/projects/clab/tnief/entangled-tokens/models")
if not ARTIFACTS_DIR.exists():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

reference_model = Model(id="meta-llama/Llama-3.1-8B-Instruct", type="open_source")


def build_ft_job(seed: int, hf_model_name: str, local_output_dir: str | None = None):
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,  # Reduced for 8B model
        gradient_accumulation_steps=8,  # Increased to compensate
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
        local_output_dir=local_output_dir,
    )


# Finetuning job configs - save locally
owl_ft_job = build_ft_job(
    seed=1,
    hf_model_name="llama_3.1_8b-owl_numbers",
    local_output_dir=str(ARTIFACTS_DIR / "llama_3.1_8b-owl_numbers"),
)

# Alternative: push to HuggingFace Hub (set local_output_dir=None)
owl_ft_job_hf = build_ft_job(
    seed=1, hf_model_name="llama_3.1_8b-owl_numbers", local_output_dir=None
)
