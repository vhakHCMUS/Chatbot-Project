import os
import streamlit as st
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyB9YUhHG_R9gOBdotHu0zkJsw3KK1DEiCo")

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

# Streamlit interface setup
st.title("AI Chatbot Interface")
st.write("Choose the model for your questions: Qanda or Gemini")

# Sidebar for model selection
model_option = st.sidebar.selectbox("Select Model", ("Qanda", "Gemini"))

# Start chat history
chat_history = []

# Display chat interface
user_input = st.text_input("You: ", "")

if user_input:
    chat_history.append({"role": "user", "parts": [user_input]})
    
    if model_option == "Qanda":
        # Process the Q&A option
        st.write("Using Qanda model...")
        response = "This is a placeholder response for Qanda model."
        chat_history.append({"role": "model", "parts": [response]})
    elif model_option == "Gemini":
        # Process the Gemini model
        chat_session = gemini_model.start_chat(
            history=chat_history
        )
        response = chat_session.send_message(user_input).text
        chat_history.append({"role": "model", "parts": [response]})

    # Display chat history
    for message in chat_history:
        if message["role"] == "user":
            st.write(f"**You**: {message['parts'][0]}")
        elif message["role"] == "model":
            st.write(f"**Model**: {message['parts'][0]}")

