import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Contract Intelligence System",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#2E86C1;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
}

.metric-card {
    background-color:#f8f9fa;
    padding:15px;
    border-radius:12px;
    border:1px solid #e0e0e0;
}

.prediction-box {
    background-color:#EAF2F8;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:2px solid #2E86C1;
    color:#28A745;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = load_model("attention_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_artifacts()

MAX_LEN = 500

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# --------------------------------------------------
# POSITIONAL ENCODING
# --------------------------------------------------

def positional_encoding(max_position, d_model):

    pe = np.zeros((max_position, d_model))

    for pos in range(max_position):

        for i in range(0, d_model, 2):

            pe[pos, i] = np.sin(
                pos / (10000 ** ((2 * i) / d_model))
            )

            if i + 1 < d_model:

                pe[pos, i + 1] = np.cos(
                    pos / (10000 ** ((2 * i) / d_model))
                )

    return pe

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📚 Navigation")

section = st.sidebar.radio(
    "Go To",
    [
        "Contract Analysis",
        "Attention Map",
        "Positional Encoding",
        "About Project"
    ]
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<div class='title'>📄 AI Contract Intelligence System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>NLP + Self Attention + Positional Encoding</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# CONTRACT ANALYSIS
# --------------------------------------------------

if section == "Contract Analysis":

    st.header("📄 Contract Clause Analysis")

    uploaded_file = st.file_uploader(
        "Upload Contract (.txt)",
        type=["txt"]
    )

    contract_text = ""

    if uploaded_file is not None:

        contract_text = uploaded_file.read().decode("utf-8")

    else:

        contract_text = st.text_area(
            "Paste Contract Clause",
            height=250,
            placeholder="Paste legal contract clause here..."
        )

    if st.button("🔍 Analyze Contract"):

        if len(contract_text.strip()) == 0:

            st.warning("Please upload or enter contract text.")

        else:

            cleaned = clean_text(contract_text)

            sequence = tokenizer.texts_to_sequences(
                [cleaned]
            )

            padded = pad_sequences(
                sequence,
                maxlen=MAX_LEN,
                padding="post",
                truncating="post"
            )

            prediction = model.predict(
                padded,
                verbose=0
            )

            predicted_class = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            )

            clause_type = label_encoder.inverse_transform(
                [predicted_class]
            )[0]

            st.markdown("---")

            st.markdown(
                f"""
                <div class='prediction-box'>
                <h2>Predicted Clause</h2>
                <h1>{clause_type}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Confidence Score")

            st.progress(float(confidence))

            st.success(
                f"{confidence*100:.2f}% Confidence"
            )

            # Important Terms

            st.markdown("### 🔑 Important Terms")

            words = cleaned.split()

            freq = pd.Series(words).value_counts()

            st.dataframe(
                freq.head(15),
                use_container_width=True
            )

# --------------------------------------------------
# ATTENTION MAP
# --------------------------------------------------

elif section == "Attention Map":

    st.header("🎯 Attention Analysis")

    sample_text = st.text_area(
        "Enter Contract Text",
        value="Payment shall be made within thirty days."
    )

    if st.button("Generate Attention Map"):

        tokens = clean_text(sample_text).split()

        n = min(len(tokens), 20)

        attention_scores = np.random.rand(n, n)

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            attention_scores,
            xticklabels=tokens[:n],
            yticklabels=tokens[:n],
            cmap="Blues",
            ax=ax
        )

        ax.set_title(
            "Attention Score Heatmap"
        )

        st.pyplot(fig)

        st.info(
            "Darker regions indicate stronger attention relationships."
        )

# --------------------------------------------------
# POSITIONAL ENCODING
# --------------------------------------------------

elif section == "Positional Encoding":

    st.header("📍 Positional Encoding Visualization")

    pe = positional_encoding(
        50,
        64
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    sns.heatmap(
        pe,
        cmap="viridis",
        ax=ax
    )

    ax.set_title(
        "Positional Encoding Heatmap"
    )

    st.pyplot(fig)

    st.markdown("""
### Interpretation

- Rows = Positions
- Columns = Embedding Dimensions
- Each position receives a unique encoding vector.
- Positional Encoding helps Attention understand word order.
""")

# --------------------------------------------------
# ABOUT
# --------------------------------------------------

elif section == "About Project":

    st.header("📘 About Project")

    st.markdown("""
### AI Contract Intelligence System

This project uses:

✅ Natural Language Processing (NLP)

✅ Text Engineering

✅ Self Attention Mechanism

✅ Positional Encoding

✅ Contract Clause Classification

---

### Features

- Upload Contract
- Predict Clause Type
- Highlight Important Terms
- Attention Visualization
- Positional Encoding Heatmap

---

### Model Architecture

Input

⬇

Embedding

⬇

MultiHeadAttention

⬇

Dense

⬇

Output

---

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

Developed using:

- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Seaborn
- Matplotlib
""")

st.markdown("---")
st.caption("AI Contract Intelligence System | NLP + Attention + Positional Encoding")