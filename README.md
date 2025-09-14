# Nucleus: Contextual AI-powered data extraction engine.

## Overview
Nucleus is a flexible, AI-driven platform designed to automatically extract, structure, and classify information from unstructured documents across diverse formats such as images, PDFs, and text. It converts chaotic raw data into actionable insights, enabling businesses to unlock value quickly and reliably without manual bottlenecks or extensive fine-tuning.

## How Nucleus Works
Input Processing: Users provide unstructured documents (images, scanned PDFs, etc.) alongside contextual instructions describing the extraction purpose.

1. OCR & Text Extraction: Nucleus uses advanced OCR techniques to convert document pages into raw text and word bounding boxes.

2. Dynamic Schema Generation: Based on the provided context and sample extracted text, Nucleus generates a dynamic data extraction schema defining the relevant fields customised to the use case. All fields are optional by design for maximum flexibility.

3. Data Extraction & Classification: Using the dynamic schema, Nucleus accurately extracts entities and values from the text, mapping them to schema fields while handling missing or partial data gracefully.

Output: The structured extracted data, alongside annotated documents highlighting key data points, are output for downstream usage such as analytics, compliance, or further processing.

## Installation Guide:
1. Clone the repository
```bash
git clone <repository-url>
cd <repository-directory>
```
2. Set up an environment (if you want or install things globally)
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install these packages:  

    For **document_annotator.py**  
    ```
    pip install pillow pytesseract pdf2image pydantic requests openai mistralai instructor PyQt5
    brew install tesseract # MacOS User
    brew install poppler # MacOS User
    ```

