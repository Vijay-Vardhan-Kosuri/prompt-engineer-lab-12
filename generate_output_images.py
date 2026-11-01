#!/usr/bin/env python3
"""Generate JPG images of the pipeline run outputs for documentation."""

from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the captured run output
with open("run_output.txt", "r") as f:
    full_output = f.read()

# Split into logical sections for 3 images
sections = [
    {
        "title": "Output 1 – Chunking & Vector Store Creation",
        "filename": "output_01_chunks_and_vectorstore.jpg",
        "content": """RAG Data Indexing Pipeline – Climate Change Sample
======================================================================

Number of chunks created: 11
----------------------------------------
  Chunk 1: Climate change is one of the most pressing challenges facing humanity in the 21s...
  Chunk 2: such as the burning of fossil fuels, deforestation, and industrial processes tha...
  Chunk 3: The effects of climate change are already visible and are projected to intensify...
  Chunk 4: Coastal cities and low-lying island nations face increased risks of flooding and...
  Chunk 5: Ecosystems around the world are under stress. Coral reefs are bleaching due to w...
  Chunk 6: Agriculture is particularly vulnerable: changing precipitation patterns and high...
  Chunk 7: Human health is also at risk. Heat-related illnesses, the spread of vector-borne...
  Chunk 8: Vulnerable populations, including the poor, elderly, and those in developing cou...
  Chunk 9: Mitigation and adaptation strategies are essential. Transitioning to renewable e...
  Chunk 10: protecting and restoring forests, and adopting sustainable agricultural practice...
  Chunk 11: above pre-industrial levels, ideally to 1.5 degrees. Achieving these targets req...

Building FAISS vector store with HuggingFace embeddings...
Vector store ready.
"""
    },
    {
        "title": "Output 2 – Query: Effects of Climate Change",
        "filename": "output_02_query_effects.jpg",
        "content": """======================================================================
Query: What are the effects of climate change?
======================================================================
Top 2 retrieved chunks:

  Rank 1 | Relevance Score: 0.5168
  Chunk: The effects of climate change are already visible and are projected to intensify.
  Rising global temperatures have caused glaciers and polar ice caps to melt at
  unprecedented rates, contributing to sea-level rise.

  Rank 2 | Relevance Score: 0.4723
  Chunk: Climate change is one of the most pressing challenges facing humanity in the
  21st century. It refers to long-term shifts in temperatures and weather patterns,
  primarily driven by human activities.
"""
    },
    {
        "title": "Output 3 – Additional Queries (Mitigation & Health/Ecosystems)",
        "filename": "output_03_queries_mitigation_health.jpg",
        "content": """======================================================================
Query: How can we mitigate climate change?
======================================================================
Top 2 retrieved chunks:

  Rank 1 | Relevance Score: 0.4342
  Chunk: Mitigation and adaptation strategies are essential. Transitioning to renewable
  energy sources such as solar, wind, and hydroelectric power can significantly
  reduce greenhouse-gas emissions. Improving energy efficiency,

  Rank 2 | Relevance Score: 0.4096
  Chunk: Climate change is one of the most pressing challenges facing humanity in the
  21st century. It refers to long-term shifts in temperatures and weather patterns,
  primarily driven by human activities.

======================================================================
Query: Impact on human health and ecosystems
======================================================================
Top 2 retrieved chunks:

  Rank 1 | Relevance Score: 0.3209
  Chunk: Human health is also at risk. Heat-related illnesses, the spread of vector-borne
  diseases like malaria and dengue, and respiratory problems from air pollution
  linked to fossil-fuel combustion are expected to rise.

  Rank 2 | Relevance Score: 0.3199
  Chunk: Ecosystems around the world are under stress. Coral reefs are bleaching due to
  warmer ocean temperatures, forests are experiencing more wildfires, and many
  species are shifting their ranges or facing extinction.

======================================================================
Pipeline completed successfully.
======================================================================
"""
    },
]


def create_text_image(title: str, content: str, filename: str, width: int = 1100):
    # Use a monospace-like font if available, else default
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    # Prepare lines
    margin = 30
    line_height = 22
    title_height = 50

    # Wrap content
    lines = []
    for paragraph in content.strip().split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(paragraph, width=95)
            lines.extend(wrapped if wrapped else [""])

    img_height = title_height + margin + len(lines) * line_height + margin + 20
    img = Image.new("RGB", (width, img_height), color=(18, 18, 24))
    draw = ImageDraw.Draw(img)

    # Accent bar
    draw.rectangle([0, 0, 8, img_height], fill=(0, 180, 120))

    # Title
    draw.text((margin + 10, 15), title, font=font_title, fill=(0, 220, 160))

    # Body
    y = title_height + 10
    for line in lines:
        # Color coding for scores / ranks
        if "Relevance Score" in line or "Rank" in line:
            color = (100, 200, 255)
        elif line.startswith("Query:") or line.startswith("="):
            color = (255, 200, 80)
        elif "Chunk:" in line or line.strip().startswith("Chunk"):
            color = (200, 200, 210)
        else:
            color = (210, 210, 220)
        draw.text((margin + 10, y), line, font=font_body, fill=color)
        y += line_height

    # Footer
    draw.text((margin + 10, img_height - 25), "RAG Indexing Pipeline Experiment  |  Local HuggingFace Embeddings + FAISS",
              font=font_body, fill=(120, 120, 130))

    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, "JPEG", quality=95)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    for sec in sections:
        create_text_image(sec["title"], sec["content"], sec["filename"])
    print("\nAll output images generated successfully.")
