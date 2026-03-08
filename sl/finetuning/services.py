import asyncio
import random
import tempfile

import unsloth  # Must be imported before trl, transformers, peft for optimizations

from datasets import Dataset
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger
from sl.external import hf_driver, openai_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl import config
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob, OpenAIFTJob, UnslothFinetuningJob
from sl.utils import llm_utils
import torch


def dataset_row_to_chat(
    dataset_row: DatasetRow,
    use_system_prompt: bool = True,
    system_prompt: str | None = None,
    generic_prompt: str | None = None,
    prompt_prefix: str | None = None,
) -> Chat:
    """
    Convert a DatasetRow to a Chat object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings
        use_system_prompt: If True, lets tokenizer add default system prompt.
                          If False, adds an empty system message to prevent default.
        system_prompt: If set, uses this as an explicit system message.
                       Takes precedence over use_system_prompt.
        generic_prompt: If set, replaces the original prompt with this string.
        prompt_prefix: If set, prepends this text to the user message.

    Returns:
        Chat object with user message (prompt) and assistant message (completion)
    """
    messages = []
    if system_prompt is not None:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    elif not use_system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=""))
    
    prompt = generic_prompt if generic_prompt else dataset_row.prompt
    if prompt_prefix:
        prompt = f"{prompt_prefix}\n\n{prompt}"
    
    messages.extend([
        ChatMessage(role=MessageRole.user, content=prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ])
    return Chat(messages=messages)


async def _run_unsloth_finetuning_job(
    job: UnslothFinetuningJob, dataset_rows: list[DatasetRow]
) -> Model:
    source_model = job.source_model

    # Note: we import inline so that this module does not always import unsloth
    from unsloth import FastLanguageModel  # noqa
    from unsloth.trainer import SFTTrainer  # noqa

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        # TODO support not hardcoding this
        max_seq_length=2048,  # Context length
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    # Create data collator for completion-only training
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    chats = [
        dataset_row_to_chat(
            row,
            use_system_prompt=job.use_system_prompt,
            system_prompt=job.system_prompt,
            generic_prompt=job.generic_prompt,
            prompt_prefix=job.prompt_prefix,
        )
        for row in dataset_rows
    ]
    if job.system_prompt is not None:
        logger.info(f"Using custom system prompt: {job.system_prompt!r}")
    else:
        logger.info(f"Using default system prompt: {job.use_system_prompt}")
    if job.prompt_prefix:
        logger.info(f"Using prompt prefix: {job.prompt_prefix!r}")
    if job.generic_prompt:
        logger.info(f"Using generic prompt: {job.generic_prompt!r}")
    dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    train_cfg = job.train_cfg

    # Set up optimizer
    custom_optimizers = (None, None)
    if job.optimizer == "muon":
        from muon import MuonWithAuxAdam
        
        # Muon works on 2D+ params; use Adam fallback for 1D params (biases, etc.)
        muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        
        logger.info(f"Muon optimizer: {len(muon_params)} 2D+ params, {len(adam_params)} 1D params")
        
        param_groups = []
        if muon_params:
            param_groups.append(dict(params=muon_params, lr=train_cfg.lr, use_muon=True))
        if adam_params:
            param_groups.append(dict(params=adam_params, lr=train_cfg.lr, use_muon=False))
        
        optimizer = MuonWithAuxAdam(param_groups)
        custom_optimizers = (optimizer, None)
        logger.info("Using Muon optimizer with AdamW fallback for 1D params")
    else:
        logger.info("Using default AdamW optimizer")

    trainer = SFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,  # Sometimes TRL fails to load the tokenizer
        optimizers=custom_optimizers,
        args=SFTConfig(
            max_seq_length=train_cfg.max_seq_length,
            packing=False,
            output_dir=None,
            num_train_epochs=train_cfg.n_epochs,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=train_cfg.lr,
            max_grad_norm=train_cfg.max_grad_norm,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            warmup_steps=train_cfg.warmup_steps,
            seed=job.seed,
            dataset_num_proc=1,
            logging_steps=1,
            # Hardware settings
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    trainer.train()
    
    # Save locally or push to HuggingFace Hub
    if job.local_output_dir:
        logger.info(f"Saving model locally to {job.local_output_dir}")
        id = hf_driver.save_local(job.local_output_dir, model, tokenizer)
        from sl.utils.file_utils import save_json
        config_path = f"{job.local_output_dir}/ft_config.json"
        save_json(job, config_path)
        logger.info(f"Saved finetuning config to {config_path}")
    else:
        logger.info(f"Pushing model to HuggingFace Hub as {job.hf_model_name}")
        id = hf_driver.push(job.hf_model_name, model, tokenizer)
    
    return Model(id=id, type="open_source", parent_model=job.source_model)


async def _run_openai_finetuning_job(
    cfg: OpenAIFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model.id}")

    prompts = [dataset_row_to_chat(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for prompt in prompts:
            f.write((prompt.model_dump_json() + "\n").encode())
        for prompt in prompts:
            # Convert Chat to OpenAI format
            f.write((prompt.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Finetuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Finetuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Finetuning job {oai_job.id} was cancelled")
            raise RuntimeError("Finetuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)
    assert oai_job.fine_tuned_model is not None
    return Model(id=oai_job.fine_tuned_model, type="openai")


async def run_finetuning_job(job: FTJob, dataset: list[DatasetRow]) -> Model:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        job: Finetuning configuration
        dataset: List of dataset rows to use for training

    Raises:
        NotImplementedError: If the model type is not supported
    """

    logger.info(
        f"Starting fine-tuning job for {job.source_model.type} model: {job.source_model.id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(job, OpenAIFTJob):
        model = await _run_openai_finetuning_job(job, dataset)
    if isinstance(job, UnslothFinetuningJob):
        model = await _run_unsloth_finetuning_job(job, dataset)
    else:
        raise NotImplementedError(
            f"Finetuning for model type '{job.source_model.type}' is not implemented"
        )

    logger.success(f"Finetuning job completed successfully! External ID: {model.id}")
    return model
