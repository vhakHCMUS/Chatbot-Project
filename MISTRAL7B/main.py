import yaml
import logging
from dataset import load_config, load_dataset_and_tokenizer, preprocess_function
from model import load_model
from trainer_setup import setup_training
from utils import setup_logging, check_device

# Load and parse configuration
config_str = open('config.yaml').read()
config = load_config(config_str)

# Set up logging
logger = setup_logging()

# Check device availability
device = check_device()
logger.info(f"Using device: {device}")

# Load dataset and tokenizer
ds, tokenizer = load_dataset_and_tokenizer(config)

# Preprocess dataset
logger.info(f"Example from the dataset: {ds['train'][0]}")
tokenized_ds = ds.map(lambda x: preprocess_function(x, config, tokenizer), batched=True)
logger.info(f"Example from the tokenized dataset: {tokenized_ds['train'][0]}")

# Load model
model_name = config['base_model']
model = load_model(model_name, device)

# Set up and train the model
trainer = setup_training(config, tokenized_ds, model, tokenizer)
trainer.train()

# Save the model and tokenizer
model.save_pretrained("results")
tokenizer.save_pretrained("results")
logger.info("Model and tokenizer saved successfully.")
