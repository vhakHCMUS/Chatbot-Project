import os
import streamlit as st
import zipfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from safetensors.torch import load_file  # Ensure you have safetensors installed

# Unzip the checkpoint if it's not unzipped yet
checkpoint_dir = "./fine-tuned-model/checkpoint-2500"  # Replace with your checkpoint directory
checkpoint_zip = "./fine-tuned-model/checkpoint-2500.zip"  # Replace with your checkpoint zip file

if not os.path.exists(checkpoint_dir):
    with zipfile.ZipFile(checkpoint_zip, 'r') as zip_ref:
        zip_ref.extractall(checkpoint_dir)

# Streamlit session state to store model and tokenizer
if "model" not in st.session_state:
    st.write("Loading model for the first time...")

    # Initialize Accelerator
    accelerator = Accelerator()

    base_model_id = "meta-math/MetaMath-Mistral-7B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load fine-tuned weights
    adapter_model_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    adapter_model_weights = load_file(adapter_model_path)

    # Update the model with fine-tuned weights
    model.load_state_dict(adapter_model_weights, strict=False)

    # Use accelerator to handle device placement (GPU or CPU)
    model = accelerator.prepare(model)

    # Cache model and tokenizer in session state
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
else:
    st.write("Using cached model...")
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

# Streamlit interface setup
st.title("AI Chatbot Interface")
st.write("Choose the model for your questions: Qanda or Math Solver (Vietnamese Math)")

# Sidebar for model selection
model_option = st.sidebar.selectbox("Select Model", ("Qanda", "Math Solver"))

# Start chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You: ", "")

def solve_math_problem(input_text):
    # Tokenize the input and get model's response
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    # Send input to the same device the model is on
    inputs = inputs.to(model.device)
    outputs = model.generate(inputs, max_length=1024, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if user_input:
    st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
    
    if model_option == "Qanda":
        # Process the Q&A option
        st.write("Using Qanda model...")
        response = "This is a placeholder response for Qanda model."
        st.session_state.chat_history.append({"role": "model", "parts": [response]})
    
    elif model_option == "Math Solver":
        # Process the Math Solver option using the fine-tuned model
        st.write("Using Math Solver model...")
        response = solve_math_problem(user_input)
        st.session_state.chat_history.append({"role": "model", "parts": [response]})

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**You**: {message['parts'][0]}")
        elif message["role"] == "model":
            st.write(f"**Model**: {message['parts'][0]}")
