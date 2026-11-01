#!/usr/bin/env python3
"""
Basic Data Indexing Pipeline for RAG
------------------------------------
Loads a sample text about climate change, chunks it, embeds it with a local
HuggingFace model, stores in FAISS, and performs similarity search.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# 1. Sample text (hardcoded – can be replaced by loading from a file)
# ---------------------------------------------------------------------------
SAMPLE_TEXT = """
Climate change is one of the most pressing challenges facing humanity in the 21st century.
It refers to long-term shifts in temperatures and weather patterns, primarily driven by human activities
such as the burning of fossil fuels, deforestation, and industrial processes that release greenhouse gases
into the atmosphere. These gases, including carbon dioxide, methane, and nitrous oxide, trap heat and
cause the greenhouse effect, leading to a gradual warming of the planet.

The effects of climate change are already visible and are projected to intensify. Rising global temperatures
have caused glaciers and polar ice caps to melt at unprecedented rates, contributing to sea-level rise.
Coastal cities and low-lying island nations face increased risks of flooding and permanent inundation.
Extreme weather events such as hurricanes, droughts, heatwaves, and heavy rainfall have become more frequent
and severe, disrupting agriculture, infrastructure, and human lives.

Ecosystems around the world are under stress. Coral reefs are bleaching due to warmer ocean temperatures,
forests are experiencing more wildfires, and many species are shifting their ranges or facing extinction.
Agriculture is particularly vulnerable: changing precipitation patterns and higher temperatures can reduce
crop yields, threaten food security, and increase the prevalence of pests and diseases.

Human health is also at risk. Heat-related illnesses, the spread of vector-borne diseases like malaria and dengue,
and respiratory problems from air pollution linked to fossil-fuel combustion are expected to rise.
Vulnerable populations, including the poor, elderly, and those in developing countries, will bear a
disproportionate burden of these impacts.

Mitigation and adaptation strategies are essential. Transitioning to renewable energy sources such as solar,
wind, and hydroelectric power can significantly reduce greenhouse-gas emissions. Improving energy efficiency,
protecting and restoring forests, and adopting sustainable agricultural practices also play crucial roles.
International agreements like the Paris Agreement aim to limit global warming to well below 2 degrees Celsius
above pre-industrial levels, ideally to 1.5 degrees. Achieving these targets requires coordinated action by
governments, businesses, and individuals worldwide.
"""

# ---------------------------------------------------------------------------
# 2. Chunk the text
# ---------------------------------------------------------------------------
def create_chunks(text: str, chunk_size: int = 300, chunk_overlap: int = 30):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": "climate_change_sample"}) for chunk in chunks]
    return documents

# ---------------------------------------------------------------------------
# 3. Create embeddings & FAISS vector store
# ---------------------------------------------------------------------------
def build_vectorstore(documents):
    # Local embedding model – no API key required
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# ---------------------------------------------------------------------------
# 4. Similarity search
# ---------------------------------------------------------------------------
def retrieve(vectorstore, query: str, k: int = 2):
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("RAG Data Indexing Pipeline – Climate Change Sample")
    print("=" * 70)

    # Step 1-2: Create chunks
    documents = create_chunks(SAMPLE_TEXT, chunk_size=300, chunk_overlap=30)
    print(f"\nNumber of chunks created: {len(documents)}")
    print("-" * 40)
    for i, doc in enumerate(documents):
        preview = doc.page_content[:80].replace("\n", " ") + "..."
        print(f"  Chunk {i+1}: {preview}")

    # Step 3-4: Embeddings + FAISS
    print("\nBuilding FAISS vector store with HuggingFace embeddings...")
    vectorstore = build_vectorstore(documents)
    print("Vector store ready.\n")

    # Step 5-6: Query & retrieve
    queries = [
        "What are the effects of climate change?",
        "How can we mitigate climate change?",
        "Impact on human health and ecosystems",
    ]

    for query in queries:
        print("=" * 70)
        print(f"Query: {query}")
        print("=" * 70)
        results = retrieve(vectorstore, query, k=2)
        print(f"Top {len(results)} retrieved chunks:\n")
        for rank, (doc, score) in enumerate(results, 1):
            print(f"  Rank {rank} | Relevance Score: {score:.4f}")
            print(f"  Chunk: {doc.page_content.strip()}")
            print()

    print("=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
