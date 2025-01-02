import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from safetensors.torch import load_file  # Ensure you have safetensors installed
from dotenv import load_dotenv
import shelve
import time
import re

# Configure Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
base_model = "meta-math/MetaMath-Mistral-7B"

st.set_page_config(
        page_title='Math Chatbot',
        page_icon="🤖"                  
        )

# # Helper functions
# def load_math_solver_model(checkpoint_dir):
#     base_model_id = base_model
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
#     tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
#     tokenizer.pad_token = tokenizer.eos_token

#     adapter_model_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
#     adapter_model_weights = load_file(adapter_model_path)
#     model.load_state_dict(adapter_model_weights, strict=False)

#     return model, tokenizer

def gemini_image_to_text(image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(["Chỉ gửi nội dung của ảnh", image])
    return response.text

def gemini_eng_to_vi(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(["Nếu input text là tiếng anh thì dịch toàn bộ qua tiếng việt, ngược lại trả lại input text", text])
    return response.text

def gemini_vi_to_eng(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(["Nếu input text là tiếng việt thì dịch toàn bộ qua tiếng anh, ngược lại trả lại input text", text])
    return response.text

def solve_math_problem(model, tokenizer, input_text):
    """Solve the math problem using the specified model and tokenizer."""
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=1024, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

checkpoint_dir = "./math-eng/checkpoint-984"  

if "model" not in st.session_state:
    st.write("Loading model for the first time...")

    accelerator = Accelerator()

    base_model_id = base_model
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # # Load base model
    # model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16, # change to torch.float16 if you're using V100
        device_map="auto",
        use_cache=True,
    )

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     base_model_id,
    #     padding_side="left",
    #     add_eos_token=True,
    #     add_bos_token=True,
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # # Load fine-tuned weights
    adapter_model_path = checkpoint_dir + "/adapter_model.safetensors"
    adapter_model_weights = load_file(adapter_model_path)

    # # Update the model with fine-tuned weights
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


# Load models
#st.write("Loading models...")
#math_solver_v1, tokenizer_v1 = load_math_solver_model("./checkpoint-984")
#math_solver_v2, tokenizer_v2 = load_math_solver_model("./eng-meta-math")

def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()


model_option = st.sidebar.selectbox("Chọn Model", ("Mistral-7B + Vietnamese-meta-math-MetaMathQA + THCS (VI)", "Mistral-7B + THCS (ENG)", "Vistral-7B + THCS (VI)"))

with st.sidebar:
    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = []
        save_chat_history([])


# Streamlit interface
st.title("Chatbot AI với Math Model và OCR")

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("Nhập vào câu hỏi?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""

        user_message = st.session_state["messages"][-1]["content"]

        try:
            pre_user_message = gemini_eng_to_vi(user_message)
            raw_response = gemini_eng_to_vi(solve_math_problem(model, tokenizer, (pre_user_message)))
            #raw_response = solve_math_problem(model, tokenizer, (user_message))
            for char in raw_response:
                full_response += char
                message_placeholder.markdown(full_response + "|")
                time.sleep(0.001) 

            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
            message_placeholder.markdown(full_response)

        st.session_state["messages"].append({"role": "assistant", "content": full_response})

st.divider()

if "form_key" not in st.session_state:
    st.session_state.form_key = 0

if "image_just_processed" not in st.session_state:
    st.session_state.image_just_processed = False

form_container = st.empty()

with form_container.form(f"upload_form_{st.session_state.form_key}"):
    uploaded_file = st.file_uploader("Tải lên một ảnh:", type=["jpg", "jpeg", "png"])
    submit_button = st.form_submit_button("Xử lý ảnh")
    
    if submit_button and uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        st.session_state["messages"].append({"role": "user", "content": "Người dùng đã tải lên một ảnh."})
        
        with st.spinner("Đang xử lý hình ảnh..."):
            try:
                ocr_text = gemini_image_to_text(image)
                ocr_text = re.sub(r"^Đây là nội dung( của)? ảnh:\s*", "", ocr_text)
                
                st.session_state.messages.append({"role": "user", "content": ocr_text})
                
                #raw_response = gemini_vi_to_eng(ocr_text)
                #raw_response = solve_math_problem(model, tokenizer, (ocr_text))
                pre_user_message = gemini_eng_to_vi(ocr_text)
                raw_response = gemini_eng_to_vi(solve_math_problem(model, tokenizer, (pre_user_message)))
                st.session_state["messages"].append({"role": "assistant", "content": raw_response})
                
                st.session_state.image_just_processed = True
                
                st.session_state.form_key += 1
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra với OCR: {e}")
            finally:
                if uploaded_file:
                    uploaded_file.close()

if st.session_state.image_just_processed:
    st.session_state.image_just_processed = False
    st.rerun()

save_chat_history(st.session_state.messages)