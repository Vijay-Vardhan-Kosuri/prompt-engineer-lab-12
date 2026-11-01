# RAG Data Indexing Pipeline Experiment

## Aim
Build a basic data indexing pipeline for Retrieval-Augmented Generation (RAG): load a sample text document, chunk it, embed it, store it in a FAISS vector store, and test retrieval with similarity search.

## Features
- Uses a local HuggingFace embedding model (`all-MiniLM-L6-v2`) – **no OpenAI API key required**.
- RecursiveCharacterTextSplitter (chunk size 300, overlap 30).
- FAISS vector store for efficient similarity search.
- Demonstrates retrieval of top-2 most relevant chunks for multiple queries.

## Requirements
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python rag_indexing_pipeline.py
```

## Expected Output
- Number of chunks created from the sample climate-change text.
- For each query, the top-2 retrieved chunks with relevance scores.

## Sample Queries Included
1. "What are the effects of climate change?"
2. "How can we mitigate climate change?"
3. "Impact on human health and ecosystems"

## Notes
- The script is fully offline after the first model download.
- You can replace the hardcoded `SAMPLE_TEXT` with a file load if desired.
- Outputs of runs are saved as images in the `output/` folder for documentation.
