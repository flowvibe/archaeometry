import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
from pinecone_client import Pinecone

# Load environment
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# Initialize model
model = SentenceTransformer("intfloat/multilingual-e5-large")

# App layout
st.set_page_config(page_title="Archaeometry", layout="centered")
st.markdown("<h1 style='color:#333333;'>Archaeometry</h1>", unsafe_allow_html=True)
query = st.text_input("Ask a question", placeholder="e.g., How did Thales predict solar eclipses?")

if query:
    with st.spinner("Thinking..."):
        # Embed the query
        query_embedding = model.encode(query).tolist()

        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        # Extract context
        context_chunks = []
        citations = []
        for match in results["matches"]:
            metadata = match["metadata"]
            text = metadata.get("text", "[No text found]").strip()
            source = metadata.get("source", "?")
            page = int(float(metadata.get("page", 0)))  # Clean up decimals

            if text:
                context_chunks.append(f"(Source: {source}, Page: {page})\n{text}")
                citations.append(f"{source}, page {page}")

        # Compose the prompt
        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"""You are a helpful assistant answering based on historical texts and scientific documents.

Answer the following question using only the information in the provided context.
If the answer cannot be found, say \"I don't know\" instead of making something up.

### Context:
{context}

### Question:
{query}

### Answer:"""

        # Get completion
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = response.choices[0].message["content"].strip()

    # Show results
    st.markdown("**Answer:**")
    st.markdown(answer)

    # Show citations only if the model gave a real answer
    if not answer.lower().startswith("i don't know") and citations:
        st.markdown("**Citations:**")
        for cite in citations:
            st.markdown(f"- {cite}")




