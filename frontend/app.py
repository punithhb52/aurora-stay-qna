import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")

st.set_page_config(page_title="AuroraStays", page_icon="🏨", layout="wide")

with st.sidebar:
    st.header("⚙️ Parameters")
    k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=3)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    max_tokens = st.slider("Maximum answer tokens", min_value=64, max_value=2048, value=512, step=64)

st.title("🏨 AuroraStays")
st.write("Ask a question about AuroraStays hotel and property policies.")
response_type = st.radio(
    "Response mode", ["Answer only", "Answer + Relevant Chunks"], horizontal=True
)
query = st.text_input(
    "Your question", placeholder="e.g., What is the pet fee at ASH-CHI Loop?"
)

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        endpoint = "/v1/answer" if response_type == "Answer only" else "/v1/answer_with_relevant_chunks"
        payload = {
            "query": query,
            "k": k,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            with st.spinner("Checking the hotel policy documents..."):
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])
                if response_type == "Answer + Relevant Chunks":
                    st.subheader("Relevant policy excerpts")
                    for i, chunk in enumerate(result.get("relevant_chunks", []), start=1):
                        with st.expander(f"Source excerpt {i}"):
                            st.write(chunk)
            else:
                st.error(f"The RAG API returned status {response.status_code}.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Unable to connect to the RAG API at {BACKEND_URL}: {exc}")
