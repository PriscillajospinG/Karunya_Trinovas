import streamlit as st
from openvino.runtime import Core
import numpy as np
from transformers import AutoTokenizer
import os

# 🧠 Optimization settings
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENVINO_NUM_STREAMS"] = "1"

# 💬 Prompt template for LLaMA 2
def build_prompt(user_input):
    return f"""<s>[INST] <<SYS>>
You are a helpful, friendly assistant.
<</SYS>>

{user_input.strip()} [/INST]"""

# ⚙️ Load tokenizer (LLaMA 2)
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        r"C:\Users\joshl\PycharmProjects\pythonProject\.venv\openvino_model\pytorch_model",
        use_fast=False
    )

# ⚙️ Load OpenVINO compiled model
@st.cache_resource
def load_model():
    ie = Core()
    compiled_model = ie.compile_model(
        r"C:\Users\joshl\PycharmProjects\pythonProject\.venv\openvino_model\model.xml", "AUTO"
    )
    return compiled_model

# 🔁 Generate response
def generate_response(prompt, tokenizer, compiled_model, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    generated_ids = input_ids.copy()

    input_layer_ids = compiled_model.input(0)
    attention_mask_layer = compiled_model.input(1)
    output_layer = compiled_model.output(0)

    response_text = ""

    for _ in range(max_tokens):
        outputs = compiled_model([generated_ids, attention_mask])
        logits = outputs[output_layer]
        next_token_id = int(np.argmax(logits[0, -1]))

        if next_token_id in tokenizer.all_special_ids:
            break

        token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        response_text += token_text

        generated_ids = np.append(generated_ids, [[next_token_id]], axis=1)
        attention_mask = np.append(attention_mask, [[1]], axis=1)

    return response_text.strip()


# 🔵 Streamlit Interface
st.set_page_config(page_title="LLaMA 2 Chatbot", page_icon="🦙")
st.title("🦙 LLaMA 2 Chatbot (OpenVINO-INT8)")
st.markdown("Ask anything! Powered by OpenVINO + LLaMA 2 + Streamlit.")

# Load models
tokenizer = load_tokenizer()
compiled_model = load_model()

# Chat interaction
user_input = st.text_input("You:", placeholder="Ask me something...")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking... 🤔"):
        prompt = build_prompt(user_input)
        response = generate_response(prompt, tokenizer, compiled_model)
        st.success("🤖 " + response)
