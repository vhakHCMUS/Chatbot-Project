import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import yaml

def load_config(config_str):
    return yaml.safe_load(config_str)

def load_dataset_and_tokenizer(config):
    # Load dataset
    try:
        ds = load_dataset("5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated")
        logging.info("Dataset loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
    
    # Load tokenizer
    model_name = config['base_model']
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    return ds, tokenizer

def preprocess_function(examples, config, tokenizer):
    max_seq_length = config['input_features'][0]['preprocessing']['max_sequence_length']

    inputs = [
        config["prompt"]["template"].format(
            instruction=instruction,
            input=example_input
        )
        for instruction, example_input in zip(examples["instruction"], examples["input"])
    ]
    model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True, padding="max_length")
    
    labels = tokenizer(
        examples["output"], max_length=max_seq_length, truncation=True, padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
