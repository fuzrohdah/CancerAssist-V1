# CancerAssist
**An AI-Driven Platform for Breast Cancer & Melanoma Analysis**


CancerAssist is an AI-powered medical analysis platform with three core modules:  
1. **OncoBot** – a knowledge-based chatbot for breast cancer and melanoma Q&A  
2. **OncoScanSkin** – a CNN-based real-time skin lesion classifier  
3. **OncoTypeBC** – a BRCA subtype predictor using multi-omic input data  

---

## 🔍 Modules Overview

### 🧠 OncoBot — RAG Medical Chatbot
A Retrieval-Augmented Generation (RAG) assistant that answers user queries on breast cancer and melanoma using trusted sources.

- **Embeddings**: MiniLM (SentenceTransformers)
- **Retrieval**: FAISS dense vector search
- **LLM**: Mistral (locally or via API)
- **Sources**: NCCN, NIH, Mayo Clinic, ClinicalTrials.gov, PubMed, FDA
- **Output**: Accurate, layman-friendly answers with context retention

### 🩺 OncoScanSkin — Skin Lesion Classifier
A computer vision model for melanoma screening via dermoscopic image classification.

- **Model**: CNN trained on benign/malignant lesion dataset
- **Input**: Uploaded image or webcam feed
- **Output**: Real-time prediction with optional Grad-CAM overlay

### 🔬 OncoTypeBC — Breast Cancer Subtype Predictor
Classifies BRCA subtypes using a Random Forest model trained on TCGA-BRCA multi-omic data (`n=945`).

- **Input**: `.xlsx` file with 20 gene expressions, 5 mutation flags, and 2 CNV amplifications (`MYC`, `CCND1`)
- **Output**: Subtype (Basal, Her2, LumA, LumB) with confidence
- **Requirement**: Strict input format matching training schema

---

## 📁 Input Format for OncoTypeBC

Your Excel file **must include exactly**:

- `20` gene expression columns (normalized)
- `5` binary mutation flags (0 or 1)
- `2` binary CNV flags: `MYC_amp`, `CCND1_amp`

Total: `27 columns`, 1 row per sample.  
Incorrect formats will raise validation errors.

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/CancerAssist.git
cd CancerAssist
pip install -r requirements.txt
