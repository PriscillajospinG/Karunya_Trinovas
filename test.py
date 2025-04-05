from openvino.runtime import Core
import numpy as np
from transformers import AutoTokenizer
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Adjust based on your cores
os.environ["OPENVINO_NUM_STREAMS"] = "1"

# Load tokenizer (LLaMA 2 style tokenizer)
tokenizer = AutoTokenizer.from_pretrained(
    r"C:\Users\joshl\PycharmProjects\pythonProject\.venv\openvino_model\pytorch_model",
    use_fast=False  # Use fast tokenizer only if it works properly with your model
)

# Load OpenVINO model
ie = Core()
compiled_model = ie.compile_model(
    r"C:\Users\joshl\PycharmProjects\pythonProject\.venv\openvino_model\model.xml", "AUTO"
)

# Get model input/output
input_ids_layer = compiled_model.input(0)
attention_mask_layer = compiled_model.input(1)
output_layer = compiled_model.output(0)

# Prompt template for LLaMA 2
def build_prompt(user_input):
    return f"""<s>[INST] <<SYS>>
You are a helpful, friendly assistant.
<</SYS>>

{user_input.strip()} [/INST]"""

print("🤖 LLaMA 2 Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("> You: ").strip()
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Format the prompt
    prompt = build_prompt(user_input)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    generated_ids = input_ids.copy()

    print("🤖 ", end="", flush=True)

    for _ in range(100):  # Generate up to 100 tokens
        # Run inference
        outputs = compiled_model([generated_ids, attention_mask])
        logits = outputs[output_layer]
        next_token_id = int(np.argmax(logits[0, -1]))

        # Stop generation at special tokens
        if next_token_id in tokenizer.all_special_ids:
            break

        # Decode next token and stream it
        token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        print(token_text, end="", flush=True)

        # Append token for next step
        generated_ids = np.append(generated_ids, [[next_token_id]], axis=1)
        attention_mask = np.append(attention_mask, [[1]], axis=1)

    print()
