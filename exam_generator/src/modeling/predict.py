from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "DeepSeek-R1-Distill-Qwen-1.5B-Medical-Reasoning",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
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

    base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    lora_adapter_id = "kingabzpro/DeepSeek-R1-Distill-Qwen-1.5B-Medical-Reasoning"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_id,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # Example inference
    prompt = """
Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>. Write the answer in between <answer></answer>.

### Question:
A research group wants to assess the relationship between childhood diet and cardiovascular disease in adulthood.\nA prospective cohort study of 500 children between 10 to 15 years of age is conducted in which the participants' diets are recorded for 1 year and then the patients are assessed 20 years later for the presence of cardiovascular disease.\nA statistically significant association is found between childhood consumption of vegetables and decreased risk of hyperlipidemia and exercise tolerance.\nWhen these findings are submitted to a scientific journal, a peer reviewer comments that the researchers did not discuss the study's validity.\nWhich of the following additional analyses would most likely address the concerns about this study's design?\n{'A': 'Blinding', 'B': 'Crossover', 'C': 'Matching', 'D': 'Stratification', 'E': 'Randomization'},\n### Response:\n<analysis>\n"""
    inputs = tokenizer(
        [prompt + tokenizer.eos_token],
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logger.info(response[0].split("### Response:")[1])
    # -----------------------------------------


if __name__ == "__main__":
    app()
