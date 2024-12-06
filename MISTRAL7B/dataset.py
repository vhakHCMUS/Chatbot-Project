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
    
    # Set up tokenizer with padding
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
    tokenizer.padding_side = "right"  # Pad on the right side
    
    logging.info("Tokenizer loaded successfully.")
    return ds, tokenizer

def preprocess_function(examples, config, tokenizer):
    max_length = config.get('max_length', 2048)
    
    # Format each example as a conversation
    conversations = []
    for query_en, response_en in zip(examples["query_en"], examples["response_en"]):
        # Format: simple Q&A format
        conversation = f"Question: {query_en}\nAnswer: {response_en}"
        conversations.append(conversation)
    
    # Tokenize the text
    tokenized_inputs = tokenizer(
        conversations,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors="pt"
    )
    
    # Set up labels for training
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs
