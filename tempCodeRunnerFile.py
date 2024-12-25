# Load the model and tokenizer
model_name = "meta-math/MetaMath-Mistral-7B"  # Assuming your model is a version of this
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)