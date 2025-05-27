from pathlib import Path

import streamlit as st
import torch
from transformers import MarianTokenizer, GenerationConfig

from marian_big_architecture import MarianMTModel

# ───────────────────────── Config ────────────────────────────
MODEL_DIR = Path("finetuned_en_el")  # ← adjust to wherever you saved checkpoints

# ───────────────────────── Loading ───────────────────────────
@st.cache_resource(show_spinner="Loading model… (first run may take a minute)")
def load_model(path: Path):
    tok = MarianTokenizer.from_pretrained(path, use_fast=True)
    mdl = MarianMTModel.from_pretrained(path)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model(MODEL_DIR)

# ───────────────────────── UI ────────────────────────────────
st.set_page_config(page_title="English→Greek Translator", layout="centered")
st.title("🇬🇧 → 🇬🇷 Translator (fine‑tuned)")

text = st.text_area("Enter English text", height=150)

with st.sidebar:
    st.header("Decoding settings")
    beam  = st.slider("Beam width",     1, 12, 6)
    max_l = st.slider("Max tokens",    10, 128, 64)
    temp  = st.slider("Temperature",  0.7, 1.3, 1.0, 0.1)
    top_k = st.slider("top‑k",          0, 100, 0)
    top_p = st.slider("top‑p",         0.0, 1.0, 1.0, 0.05)

if st.button("Translate") and text.strip():
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    gen_cfg = GenerationConfig(
        max_length=max_l,
        num_beams=beam,
        temperature=temp,
        top_k=top_k if top_k else None,
        top_p=top_p,
        do_sample=(temp != 1.0 or top_k > 0 or top_p < 1.0),
    )

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_cfg.to_dict())

    greek = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    st.success(greek)
