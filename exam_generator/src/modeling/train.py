from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
import gc

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "DeepSeek-R1-Distill-Qwen-1.5B-Medical-Reasoning",
    # -----------------------------------------
):
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)
        logger.info("✅ Hugging Face login successful.")
    else:
        logger.error("❌ HF_TOKEN not found. Please check your .env file.")
        return

    model_dir = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    train_prompt_style = """
Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>. Write the answer in between <answer></answer>.
### Question:
{}

### Response:
{}"""
    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for question, response in zip(inputs, outputs):
            question = question.replace("Q:", "")
            if not response.endswith(tokenizer.eos_token):
                response += tokenizer.eos_token
            text = train_prompt_style.format(question, response)
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset(
        "mamachang/medical-reasoning",
        split="train",
        trust_remote_code=True,
    )
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    training_arguments = TrainingArguments(
        output_dir=str(model_path),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="tensorboard"
    )
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = False
    logger.info("Starting training...")
    trainer.train()
    logger.success("Modeling training complete.")
    # Save model to Hugging Face Hub
    new_model_name = "DeepSeek-R1-Distill-Qwen-1.5B-Medical-Reasoning"
    trainer.model.push_to_hub(new_model_name)
    trainer.processing_class.push_to_hub(new_model_name)
    # -----------------------------------------


if __name__ == "__main__":
    app()
