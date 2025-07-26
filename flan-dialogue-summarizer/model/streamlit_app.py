import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("model")
    tokenizer = AutoTokenizer.from_pretrained("model")
    return model, tokenizer

model, tokenizer = load_model()

st.title("🧠 Dialogue Summarizer with FLAN-T5")
st.write("Enter a human conversation below, and it will be summarized using Google's FLAN-T5 model.")

dialogue = st.text_area("Enter Dialogue:", height=200, placeholder="#Person1#: I’m late for the train.\n#Person2#: You still have time, the station is nearby.")

if st.button("Summarize"):
    if dialogue.strip() == "":
        st.warning("Please enter some dialogue.")
    else:
        prompt = f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success("✅ Summary:")
        st.text_area("Result:", summary, height=100)
