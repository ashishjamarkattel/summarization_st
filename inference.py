import streamlit as st 
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

st.write("""
# Summarize Your Text
""")

tokenizer = T5Tokenizer.from_pretrained("ashishkat/summarization")
model = T5ForConditionalGeneration.from_pretrained("ashishkat/summarization", return_dict=True)


text_input = st.text_area("Text to summarize")

if text_input:

    tokenized_text = tokenizer.encode_plus(
        str(text_input),
        return_attention_mask = True, 
        return_tensors="pt")

    generated_token = model.generate(
        input_ids = tokenized_text["input_ids"],
        attention_mask=tokenized_text["attention_mask"],
        max_length = 256,
        use_cache=True,
         )

    pred = [
           tokenizer.decode(token_ids=ids, skip_special_tokens=True) for ids in generated_token
   ]

    st.write("## Summarized Text")
    st.write(" ".join(pred))