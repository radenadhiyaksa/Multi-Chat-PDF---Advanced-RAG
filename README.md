# 📄 Advanced RAG PDF AI Analyzer

Advanced RAG (Retrieval-Augmented Generation) PDF Analyzer menggunakan:

- ⚡ Groq API
- 🧠 Llama 3.3 70B
- 🔎 FAISS Vector Search
- 📚 Semantic Retrieval
- 📷 Mistral OCR API

Project ini mampu:
- membaca PDF biasa,
- membaca PDF scan/gambar,
- melakukan semantic search,
- Advanced RAG Question Answering,
- rangkuman otomatis,
- terjemahan multi bahasa.

---

# ✨ Features

## 📝 PDF Summarization
Membuat rangkuman otomatis dokumen PDF dengan AI.

Output:
- Gambaran Umum
- Poin Penting
- Kesimpulan

---

## 💬 Advanced RAG Question Answering
Menggunakan:
- Embedding Model
- FAISS Vector Database
- Semantic Retrieval

AI hanya menjawab berdasarkan konteks hasil retrieval.

---

## 🌐 Multi-language Translation
Menerjemahkan dokumen PDF ke berbagai bahasa:
- Indonesia
- Inggris
- Melayu
- Mandarin
- Arab
- Jepang
- Prancis
- Spanyol

---

## 📷 OCR PDF Scan
Jika PDF tidak memiliki teks:
- otomatis menggunakan Mistral OCR API.

---

## 🧠 Semantic Search
Menggunakan:
- `BAAI/bge-small-en-v1.5`
- FAISS similarity search

untuk retrieval chunk paling relevan.

---

# 🧰 Tech Stack

- Python
- Groq API
- Llama 3.3 70B
- Sentence Transformers
- FAISS
- NumPy
- pdfplumber
- Mistral OCR API
- Google Colab / Jupyter Notebook

---

# 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

atau:

```bash
pip install groq pdfplumber python-dotenv httpx sentence-transformers faiss-cpu numpy
```

---

# 🔑 API Key

Project membutuhkan:

## 1. Groq API Key

Daftar:
https://console.groq.com

---

## 2. Mistral API Key

Daftar:
https://console.mistral.ai

---

# ⚙️ Environment Variables

Buat file `.env`

```env
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

---

# ▶️ Cara Menjalankan

## Python

```bash
python app_advanced_rag.py
```

---

## Google Colab

- Upload file `.ipynb`
- Jalankan cell satu per satu
- Upload PDF
- Gunakan fitur:
  - Rangkuman
  - Advanced RAG QA
  - Terjemahan

---

# 🔍 Advanced RAG Pipeline

```text
PDF
 ↓
OCR / Text Extraction
 ↓
Chunking
 ↓
Embedding
 ↓
FAISS Vector Store
 ↓
Semantic Retrieval
 ↓
Groq LLM Response
```

---

# 📂 Struktur Fitur

## 1. Upload PDF
Upload PDF langsung dari Google Colab.

---

## 2. Chunking
Dokumen dipecah menjadi beberapa chunk kecil.

---

## 3. Embedding
Setiap chunk diubah menjadi vector embedding.

---

## 4. Vector Search
Menggunakan FAISS similarity search.

---

## 5. Retrieval
Mengambil chunk paling relevan terhadap pertanyaan user.

---

## 6. LLM Generation
Groq Llama 70B menghasilkan jawaban berdasarkan retrieval.

---

# 🧠 Embedding Model

```python
BAAI/bge-small-en-v1.5
```

---

# 🚀 LLM Model

```python
llama-3.3-70b-versatile
```

Provider:
- Groq

---

# 📄 Supported File

- `.pdf`

---

# 📌 Use Cases

Cocok untuk:
- Mahasiswa
- Peneliti
- Analisis jurnal
- Semantic document search
- AI PDF assistant
- Multi PDF QA
- OCR PDF scan
- Ringkasan ebook

---

# 👨‍💻 Author

**Raden Adhiyaksa Indiharto**