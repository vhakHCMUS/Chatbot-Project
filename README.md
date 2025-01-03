# Math Chatbot with Math Model and OCR

This repository contains a Streamlit-based AI chatbot application that integrates various machine learning models to solve mathematical problems, perform OCR (Optical Character Recognition), and support multi-language translations. The chatbot leverages models such as Mistral-7B and Gemini API for enhanced user interactions.

## Features

- **Math Problem Solving:**
  - Supports solving mathematical problems using pre-trained models.
  - Fine-tuned weights for enhanced accuracy.

- **OCR Integration:**
  - Processes uploaded images to extract textual content using the Gemini API.

- **Language Translation:**
  - Translates English to Vietnamese and vice versa using Gemini API.

- **Chat History:**
  - Saves and loads chat history using `shelve` for persistent conversations.

- **Interactive User Interface:**
  - Streamlit-based UI for seamless user interaction.
  - Supports uploading images for OCR and text-based problem solving.

## Prerequisites

1. **Python 3.8+**
2. **Libraries:**
   - `streamlit`
   - `PIL` (Pillow)
   - `google.generativeai`
   - `torch`
   - `transformers`
   - `safetensors`
   - `dotenv`
   - `shelve`
   - `accelerate`
3. **Gemini API Key:**
   - Obtain an API key and store it in a `.env` file with the variable `GEMINI_API_KEY`.
4. **Checkpoints:**
   - Download fine-tuned model checkpoints from the provided Google Drive link and place them in the `./checkpoints` directory. (The link is too large to host on GitHub.)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vhakHCMUS/Chatbot-Project.git
   cd Chatbot-Project

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Set up the `.env` file:
   ```bash
   GEMINI_API_KEY=your_api_key_here

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py

2. Interact with the chatbot:
- Use the sidebar to select models or clear chat history. **Note:** Model selection is a placeholder; changing models requires manually updating the `base_model` and `checkpoint_dir` in the code.
- Upload images for OCR or type questions for the chatbot.

3. Save and load chat history automatically between sessions.

## Models and Checkpoints

### Mistral-7B
- **Checkpoint 1:** 
  - Path: `./checkpoints/math-vi/checkpoint-984`
  - Description: Fine-tuned on the Vietnamese math dataset (`math_test (vi)`).
  
- **Checkpoint 2:** 
  - Path: `./checkpoints/math-eng/checkpoint-984`
  - Description: Fine-tuned on the English math dataset (`math_test (eng)`).

### Vistral-7B
- **Checkpoint:** 
  - Path: `./checkpoints/vistral-7b/checkpoint-328`
  - Description: Fine-tuned on the Vietnamese math dataset (`math_test (vi)`).

## Project Structure

```plaintext
.
├── app.py                   # Main application code
├── checkpoints/             # Directory for model checkpoints
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md                # Project documentation
```

## Key Components

- **Chat History:**

  - Uses `shelve` to store chat history persistently.

- **Math Solver:**

  - Loads a fine-tuned model and tokenizer for solving mathematical problems.

- **OCR with Gemini:**

  - Processes uploaded images to extract and interpret textual content.

- **Translation:**

  - Utilizes Gemini API for seamless language translation between English and Vietnamese.

## Models

### Math Solver Model

- **Base Model:** Mistral-7B, Vistral-7B
- **Datasets for fine-tuning:** MetaMathQA with Vietnamese adaptations, THCS (math_test)

### Gemini API

- **Capabilities:** Image-to-text and multi-language translation

## Acknowledgments

- **Streamlit:** For providing an intuitive UI framework.
- **Hugging Face Transformers:** For pre-trained models and tokenizer utilities.
- **Gemini API:** For OCR and translation capabilities.


