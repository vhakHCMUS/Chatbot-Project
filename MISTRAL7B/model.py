from transformers import AutoModelForCausalLM
import torch

def load_model(model_name, device):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
