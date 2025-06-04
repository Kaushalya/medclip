import os
import sys

# Fix for Streamlit + PyTorch compatibility issue
import streamlit as st

# Patch the file watcher to ignore torch modules
def patched_extract_paths(module):
    """Patched version that safely handles torch modules"""
    try:
        if hasattr(module, '__name__') and 'torch' in str(module.__name__):
            return []
        if hasattr(module, '__path__'):
            return list(module.__path__)
        return []
    except:
        return []

# Apply the patch
import streamlit.watcher.local_sources_watcher as lsw
lsw.extract_paths = patched_extract_paths

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from medclip.modeling_hybrid_clip import FlaxHybridCLIP


@st.cache_resource
def load_model():
    model = FlaxHybridCLIP.from_pretrained("flax-community/medclip-roco", _do_init=True)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    return model, tokenizer

@st.cache_resource
def load_image_embeddings():
    embeddings_df = pd.read_hdf('feature_store/image_embeddings_large.hdf', key='emb')
    image_embeds = np.stack(embeddings_df['image_embedding'])
    image_files = np.asarray(embeddings_df['files'].tolist())
    return image_files, image_embeds

k = 5
img_dir = './images'

st.sidebar.header("MedCLIP")
st.sidebar.image("./assets/logo.png", width=100)
st.sidebar.empty()
st.sidebar.markdown("""Search for medical images with natural language powered by a CLIP model [[Model Card]](https://huggingface.co/flax-community/medclip-roco) finetuned on the
 [Radiology Objects in COntext (ROCO) dataset](https://github.com/razorx89/roco-dataset).""")
st.sidebar.markdown("Example queries:")
ex1_button = st.sidebar.button("🔍 pathology")
ex2_button = st.sidebar.button("🔍 ultrasound scans")
ex3_button = st.sidebar.button("🔍 pancreatic carcinoma")
ex4_button = st.sidebar.button("🔍 PET scan")

k_slider = st.sidebar.slider("Number of images", min_value=1, max_value=10, value=5)
st.sidebar.markdown("Kaushalya Madhawa, 2021")

st.title("MedCLIP 🩺")
text_value = ''
if ex1_button:
    text_value = 'pathology'
elif ex2_button:
    text_value = 'ultrasound scans'
elif ex3_button:
    text_value = 'pancreatic carcinoma'
elif ex4_button:
    text_value = 'PET scan'


image_list, image_embeddings = load_image_embeddings()
model, tokenizer = load_model()

query = st.text_input("Enter your query here:", value=text_value)
dot_prod = None

if len(query)==0:
    query = text_value

if st.button("Search") or k_slider:
    if len(query)==0:
        st.write("Please enter a valid search query")
    else:
        with st.spinner(f"Searching ROCO test set for {query}..."):
            k = k_slider
            inputs = tokenizer(text=[query], return_tensors="jax", padding=True)
            query_embedding = model.get_text_features(**inputs)
            query_embedding = np.asarray(query_embedding)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
            dot_prod = np.sum(np.multiply(query_embedding, image_embeddings), axis=1)
            topk_images = dot_prod.argsort()[-k:]
            matching_images = image_list[topk_images]
            top_scores = 1. - dot_prod[topk_images]
            #show images
            for img_path, score in zip(matching_images, top_scores):
                img = plt.imread(os.path.join(img_dir, img_path))
                st.image(img, width=300)
                st.write(f"{img_path} ({score:.2f})")
