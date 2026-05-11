# Advanced RAG PDF AI Analyzer (Python)


#!/usr/bin/env python3
"""
============================================================
  ADVANCED RAG PDF AI ANALYZER
  by Raden Adhiyaksa Indiharto
============================================================
Fitur:
- PDF Summarization
- Advanced RAG Q&A
- Translation
- OCR PDF Scan
- Vector Search (FAISS)
- Semantic Retrieval
============================================================
Install:

pip install groq pdfplumber python-dotenv httpx \
sentence-transformers faiss-cpu numpy

============================================================
"""

import os
import sys
import base64
import textwrap
import numpy as np
import faiss
import httpx

from sentence_transformers import SentenceTransformer

# ── Optional .env ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Dependency checks ─────────────────────────────────────
try:
    from groq import Groq
except ImportError:
    sys.exit("❌ Package 'groq' belum terinstall")

try:
    import pdfplumber
except ImportError:
    sys.exit("❌ Package 'pdfplumber' belum terinstall")

# ── Konstanta ─────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

MAX_TOKENS = 4096
MAX_CHARS = 100000

DIVIDER = "─" * 60

BAHASA_OPTIONS = {
    "1": "Indonesia",
    "2": "Inggris",
    "3": "Melayu",
    "4": "Mandarin (Simplified)",
    "5": "Arab",
    "6": "Jepang",
    "7": "Prancis",
    "8": "Spanyol",
}

# ── Load embedding model ──────────────────────────────────
print("📦 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)


# ══════════════════════════════════════════════════════════
# UTILITAS
# ══════════════════════════════════════════════════════════

def print_header():
    print(f"\n{'═'*60}")
    print("📄 ADVANCED RAG PDF AI ANALYZER")
    print("Semantic Search + FAISS + Groq")
    print(f"{'═'*60}\n")


def wrap_print(text, width=80):
    for para in text.split("\n"):
        if para.strip():
            print(textwrap.fill(para, width=width))
        else:
            print()


# ══════════════════════════════════════════════════════════
# API SETUP
# ══════════════════════════════════════════════════════════

def setup_api():
    key = os.environ.get("GROQ_API_KEY", "").strip()

    if not key:
        print("🔑 Masukkan Groq API Key:")
        key = input("API Key: ").strip()

    if not key:
        sys.exit("❌ API Key kosong")

    return Groq(api_key=key)


# ══════════════════════════════════════════════════════════
# OCR MISTRAL
# ══════════════════════════════════════════════════════════

def ocr_pdf_online(path):
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()

    if not api_key:
        print("🔑 Masukkan Mistral API Key:")
        api_key = input("Mistral API Key: ").strip()

    if not api_key:
        raise ValueError("Mistral API Key diperlukan")

    with open(path, "rb") as f:
        pdf_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    print("📷 OCR processing...")

    response = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
        },
        timeout=120.0,
    )

    if response.status_code != 200:
        raise ValueError(f"OCR Error: {response.text}")

    data = response.json()
    pages = data.get("pages", [])

    pages_text = []

    for page in pages:
        text = page.get("markdown", "").strip()

        if text:
            pages_text.append(
                f"[Halaman {page.get('index', 0)+1}]\n{text}"
            )

    full_text = "\n\n".join(pages_text)

    return full_text, len(pages)


# ══════════════════════════════════════════════════════════
# LOAD PDF
# ══════════════════════════════════════════════════════════

def load_pdf(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    if not path.lower().endswith(".pdf"):
        raise ValueError("File harus PDF")

    pages_text = []

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()

            if text and text.strip():
                pages_text.append(f"[Halaman {i}]\n{text.strip()}")

    if not pages_text:
        return ocr_pdf_online(path)

    full_text = "\n\n".join(pages_text)

    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS]

    return full_text, total_pages


# ══════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ══════════════════════════════════════════════════════════
# VECTOR STORE
# ══════════════════════════════════════════════════════════

def create_vector_store(chunks):
    print("🧠 Creating embeddings...")

    embeddings = embedder.encode(chunks)

    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


# ══════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════

def retrieve(query, chunks, index, top_k=5):
    query_embedding = embedder.encode([query])

    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


# ══════════════════════════════════════════════════════════
# GROQ CALL
# ══════════════════════════════════════════════════════════

def call_groq(client, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════
# FITUR RANGKUMAN
# ══════════════════════════════════════════════════════════

def fitur_rangkuman(client, pdf_text, filename):
    print(f"\n{DIVIDER}")
    print("📝 RANGKUMAN DOKUMEN")
    print(DIVIDER)

    system = (
        "Kamu adalah analis dokumen profesional. "
        "Buat rangkuman terstruktur dalam Bahasa Indonesia."
    )

    user = f"""
Dokumen:
{pdf_text[:15000]}

Buat:
1. Gambaran umum
2. Poin penting
3. Kesimpulan
"""

    result = call_groq(client, system, user)

    print()
    wrap_print(result)


# ══════════════════════════════════════════════════════════
# ADVANCED RAG Q&A
# ══════════════════════════════════════════════════════════

def fitur_tanya_jawab_rag(client, chunks, index):
    print(f"\n{DIVIDER}")
    print("💬 ADVANCED RAG QUESTION ANSWERING")
    print(DIVIDER)

    while True:
        query = input("\n❓ Pertanyaan: ").strip()

        if query.lower() in ["exit", "quit", "keluar"]:
            break

        print("\n🔍 Retrieving relevant chunks...")

        relevant_chunks = retrieve(
            query,
            chunks,
            index,
            top_k=5
        )

        context = "\n\n".join(relevant_chunks)

        system = (
            "Jawab hanya berdasarkan konteks yang diberikan. "
            "Jika informasi tidak ada, katakan tidak ditemukan. "
            "Jawab dalam Bahasa Indonesia."
        )

        user = f"""
KONTEKS:
{context}

PERTANYAAN:
{query}
"""

        answer = call_groq(client, system, user)

        print(f"\n{'─'*60}")
        print("💡 JAWABAN:\n")
        wrap_print(answer)
        print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════
# FITUR TERJEMAHAN
# ══════════════════════════════════════════════════════════

def fitur_terjemahan(client, pdf_text, filename):
    print(f"\n{DIVIDER}")
    print("🌐 TERJEMAHAN DOKUMEN")
    print(DIVIDER)

    for k, v in BAHASA_OPTIONS.items():
        print(f"[{k}] {v}")

    pilihan = input("\nPilih bahasa: ").strip()

    if pilihan not in BAHASA_OPTIONS:
        print("❌ Pilihan tidak valid")
        return

    bahasa = BAHASA_OPTIONS[pilihan]

    system = (
        f"Terjemahkan teks berikut ke Bahasa {bahasa}. "
        "Pertahankan format dan struktur."
    )

    user = pdf_text[:15000]

    result = call_groq(client, system, user)

    print()
    wrap_print(result)


# ══════════════════════════════════════════════════════════
# MENU
# ══════════════════════════════════════════════════════════

def menu(client, pdf_text, filename, total_pages, chunks, index):
    while True:
        print(f"\n{DIVIDER}")
        print(f"📄 File     : {os.path.basename(filename)}")
        print(f"📃 Halaman  : {total_pages}")
        print(f"📦 Chunks   : {len(chunks)}")
        print(f"📊 Karakter : {len(pdf_text):,}")
        print(DIVIDER)

        print("""
[1] 📝 Rangkuman
[2] 💬 Advanced RAG Q&A
[3] 🌐 Terjemahan
[4] 📂 Ganti File
[0] 🚪 Keluar
""")

        pilihan = input("Pilih: ").strip()

        if pilihan == "1":
            fitur_rangkuman(client, pdf_text, filename)

        elif pilihan == "2":
            fitur_tanya_jawab_rag(client, chunks, index)

        elif pilihan == "3":
            fitur_terjemahan(client, pdf_text, filename)

        elif pilihan == "4":
            return True

        elif pilihan == "0":
            print("👋 Bye")
            sys.exit(0)

        else:
            print("❌ Pilihan tidak valid")


# ══════════════════════════════════════════════════════════
# INPUT PDF
# ══════════════════════════════════════════════════════════

def minta_file_pdf():
    while True:
        path = input("\n📂 Path PDF: ").strip().strip('"').strip("'")

        if not path:
            continue

        try:
            print("⏳ Reading PDF...")

            pdf_text, total_pages = load_pdf(path)

            print("✅ PDF loaded")

            return path, pdf_text, total_pages

        except Exception as e:
            print(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print_header()

    client = setup_api()

    print("✅ Connected to Groq API")

    while True:
        path, pdf_text, total_pages = minta_file_pdf()

        print("🧩 Chunking document...")

        chunks = chunk_text(pdf_text)

        print(f"✅ Total chunks: {len(chunks)}")

        index, embeddings = create_vector_store(chunks)

        print("✅ Vector store created")

        ganti_file = menu(
            client,
            pdf_text,
            path,
            total_pages,
            chunks,
            index
        )

        if not ganti_file:
            break


if __name__ == "__main__":
    main()
