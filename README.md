# Intelligent Document Extraction, Validation & API Platform

A FastAPI application that accepts financial PDFs, extracts text with native PDF parsing or multi-pass Tesseract OCR, uses Groq for grounded structured extraction, validates financial totals, and stores processed results.

## Deploy on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SumanTannu/Intelligent-Document-Extraction-Validation-API-Platform)

The included Blueprint creates:

- a Docker web service containing Python, the application, and Tesseract OCR;
- a free Render Postgres database for processed-document history;
- a health check at `/api/v1/health`.

When Render asks for `LLM_API_KEY`, enter a valid Groq API key. Do not commit it to the repository. After the deploy succeeds, open the service URL for the web interface or `/docs` for the OpenAPI interface.

The free web plan has 512 MB of memory and sleeps after inactivity, so the first request after a sleep can be slow and high-resolution OCR may be constrained. The free Postgres database also expires after 30 days. For sustained use, select a paid web plan with at least 2 GB RAM and a paid database in the Render dashboard.

## Run with Docker

```bash
docker build -t intelligent-document-api .
docker run --rm -p 10000:10000 \
  -e LLM_API_KEY=your_groq_api_key \
  -e DATABASE_URL=sqlite:////tmp/intellidoc.sqlite3 \
  intelligent-document-api
```

Then open `http://localhost:10000`.

## Run locally without Docker

Install Python 3.12 and Tesseract OCR, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Configuration defaults and supported variables are listed in `.env.example`.

## API endpoints

- `GET /api/v1/health` - service health
- `POST /api/v1/documents/process` - upload and process a PDF
- `GET /api/v1/documents` - list processed documents
- `GET /api/v1/documents/{document_name}` - retrieve one result
