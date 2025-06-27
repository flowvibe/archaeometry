import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone

# Initialize Pinecone client
pc = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_ENV"]
)
index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])

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
            page = int(float(metadata.get("page", 0)))

            if text:
                context_chunks.append(f"(Source: {source}, Page: {page})\n{text}")
                citations.append(f"{source}, page {page}")

        # Compose the prompt
        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"""You are a helpful assistant answering based on historical texts and scientific documents.

Answer the following question using only the information in the provided context.
If the answer cannot be found, say "I don't know" instead of making something up.

### Context:
{context}

### Question:
{query}

### Answer:"""

        # Get completion using OpenAI v1+
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()

        # Show results
        st.markdown("**Answer:**")
        st.markdown(answer)

        # Show citations if applicable
        if not answer.lower().startswith("i don't know") and citations:
            st.markdown("**Citations:**")
            for cite in citations:
                st.markdown(f"- {cite}")

