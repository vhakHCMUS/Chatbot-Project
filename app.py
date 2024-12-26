import os
import streamlit as st
from PIL import Image
import google.generativeai as genai

# import zipfile
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from accelerate import Accelerator
# from safetensors.torch import load_file  # Ensure you have safetensors installed

# Configure the Gemini API key
genai.configure(api_key="AIzaSyATQkrOa2vZolW8bFS7K090ueFFSaXKRFY")

# Set up the model configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

# Create the GenerativeModel for Gemini
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
)

def gemini_image_to_text(image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(["Nhận diện văn bản từ ảnh", image])
    return response.text

# # Unzip the checkpoint if it's not unzipped yet
# checkpoint_dir = "./fine-tuned-model/checkpoint-2500"  # Replace with your checkpoint directory
# checkpoint_zip = "./fine-tuned-model/checkpoint-2500.zip"  # Replace with your checkpoint zip file

# if not os.path.exists(checkpoint_dir):
#     with zipfile.ZipFile(checkpoint_zip, 'r') as zip_ref:
#         zip_ref.extractall(checkpoint_dir)

# Streamlit session state to store model and tokenizer
# if "model" not in st.session_state:
#     st.write("Loading model for the first time...")

#     # Initialize Accelerator
#     accelerator = Accelerator()

#     base_model_id = "meta-math/MetaMath-Mistral-7B"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     # Load base model
#     model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_id,
#         padding_side="left",
#         add_eos_token=True,
#         add_bos_token=True,
#     )
#     tokenizer.pad_token = tokenizer.eos_token

#     # Load fine-tuned weights
#     adapter_model_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
#     adapter_model_weights = load_file(adapter_model_path)

#     # Update the model with fine-tuned weights
#     model.load_state_dict(adapter_model_weights, strict=False)

#     # Use accelerator to handle device placement (GPU or CPU)
#     model = accelerator.prepare(model)

#     # Cache model and tokenizer in session state
#     st.session_state.model = model
#     st.session_state.tokenizer = tokenizer
# else:
#     st.write("Using cached model...")
#     model = st.session_state.model
#     tokenizer = st.session_state.tokenizer

# Streamlit interface setup
st.title("AI Chatbot Interface")
st.write("Choose the model for your questions: Qanda, Math Solver, Gemini, or Gemini Image OCR")

# Sidebar for model selection
model_option = st.sidebar.selectbox("Select Model", ("Qanda", "Math Solver", "Gemini", "Gemini Image OCR"))

# Start chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You: ", "")

# def solve_math_problem(input_text):
#     # Tokenize the input and get model's response
#     inputs = tokenizer.encode(input_text, return_tensors="pt")
#     # Send input to the same device the model is on
#     inputs = inputs.to(model.device)
#     outputs = model.generate(inputs, max_length=1024, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

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
        # response = solve_math_problem(user_input)
        response = "This feature is disabled because the fine-tuned model is not available."
        st.session_state.chat_history.append({"role": "model", "parts": [response]})

    elif model_option == "Gemini":
        # Process the Gemini model
        chat_session = gemini_model.start_chat(
            history=st.session_state.chat_history
        )
        response = chat_session.send_message(user_input).text
        st.session_state.chat_history.append({"role": "model", "parts": [response]})

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**You**: {message['parts'][0]}")
        elif message["role"] == "model":
            st.write(f"**Model**: {message['parts'][0]}")

# Gemini Image OCR functionality
if model_option == "Gemini Image OCR":
    st.write("Using Gemini for Image OCR...")
    uploaded_image = st.file_uploader("Upload an image for text recognition", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing the image with Gemini..."):
            try:
                result = gemini_image_to_text(image)
                st.subheader("Recognized Text:")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
