# Automated Documentation Conversion Tool

A hackathon prototype that converts technical documentation from PDF into structured DITA XML using AI.

Built for the **BNY AI in Finance Hackathon — Automated Documentation Conversion Tool** (May 14, 2026).

## Overview

BNY's challenge was to automate the conversion of large volumes of technical documentation from PDF into clean, structured XML compliant with the DITA (Darwin Information Typing Architecture) standard.

This project implements an end-to-end prototype that:

- accepts one or more PDF documents
- extracts text, images, and external links
- identifies logical sections and document structure
- uses an LLM to classify content as DITA `concept`, `task`, or `reference`
- generates structured DITA topic files
- builds a `main.ditamap` describing topic hierarchy
- packages the generated DITA files and extracted assets into a ZIP archive

The pipeline is available both as a command-line tool and through a FastAPI web application.

## Architecture

```text
PDF files
   │
   ▼
PyMuPDF ───────────────► Text / images / links
   │
   ▼
Section & structure detection
   │
   ▼
Groq API (Llama 3.3 70B)
   │
   ▼
Structured topic data (JSON)
   │
   ▼
DITA XML generation
   │
   ├── concept.dita
   ├── task.dita
   ├── reference.dita
   ├── images/
   └── main.ditamap
   │
   ▼
ZIP package
```

## Tech Stack

- **Python**
- **FastAPI** — web API
- **PyMuPDF** — PDF parsing and image/link extraction
- **Pillow** — image processing
- **lxml** — XML/DITA generation
- **Groq API / Llama 3.3 70B** — AI-powered content structuring
- **DITA / DITA-OT** — target documentation standard and validation workflow

## Project Structure

```text
dita-xml-converter/
├── app.py
├── convert.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── static/
    └── index.html
```

Generated/runtime files such as `uploads/`, `dita_out/`, and the ZIP output are intentionally excluded from version control.

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd dita-xml-converter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file or export the variable directly:

```bash
export GROQ_API_KEY="your-api-key"
```

The application reads the key from the `GROQ_API_KEY` environment variable.

### 5. Start the web application

```bash
uvicorn app:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

## Command-Line Usage

The conversion pipeline can also be used without the web interface.

Single PDF:

```bash
python convert.py report.pdf -o dita_out
```

Multiple PDFs:

```bash
python convert.py doc1.pdf doc2.pdf doc3.pdf -o dita_out
```

Generated output includes DITA topic files and a document map.

## DITA Output

The pipeline generates three supported technical-content topic types:

- `concept` — explanatory content
- `task` — step-by-step procedures
- `reference` — tables and reference information

It also generates a DITA map containing the topic hierarchy.

The generated XML includes the DITA document types required for processing with DITA-OT.

## AI Processing

The LLM is used to transform extracted PDF content into structured topic data.

The prompt instructs the model to:

- detect topic types
- preserve the source meaning
- split content when the type changes
- generate short descriptions and keywords
- identify code blocks, tables, images, and links
- infer logical parent/child topic relationships
- avoid inventing information

The application then converts the returned structured JSON into DITA XML rather than asking the model to generate raw XML directly.

## Image Handling

Images embedded in PDFs are extracted and processed before being included in the DITA output.

The pipeline:

- extracts raster images with PyMuPDF
- resizes images to a maximum width of 1000 px
- compresses oversized images to stay within the hackathon's 200 KB target
- references the resulting files from the generated DITA topics

## Hackathon Context

The project was developed for BNY's **AI in Finance Hackathon** challenge on automated documentation conversion.

The challenge called for a prototype capable of automatically ingesting, normalizing, transforming, validating, and exporting documentation at scale, with DITA-OT used to validate the resulting DITA/XML output.

This repository contains the application code used for the prototype.

## Notes

- A Groq API key is required for AI-powered conversion.
- API usage is subject to the limits of the configured Groq account.
- DITA-OT is not bundled with this repository and can be used separately to validate/render the generated DITA content.
- The project was developed as a hackathon prototype rather than a production deployment.
