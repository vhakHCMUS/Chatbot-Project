import logging
import torch

# Set up logging
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

# Check device availability
def check_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
