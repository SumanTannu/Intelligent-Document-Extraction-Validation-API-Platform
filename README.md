# Intelligent Document Extraction, Validation & API Platform

A FastAPI application that accepts financial PDFs, extracts text with native PDF parsing or multi-pass Tesseract OCR, uses Groq for grounded structured extraction, validates financial totals, and stores processed results.

## Deploy on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SumanTannu/Intelligent-Document-Extraction-Validation-API-Platform)

The included Blueprint creates three separate resources and configures the backend health check:

- a Docker backend containing FastAPI and Tesseract OCR;
- a static frontend served independently through Render's CDN;
- a free Render Postgres database for processed-document history;

The default service URLs are:

- Frontend: `https://tannu-intellidoc-frontend.onrender.com`
- Backend: `https://tannu-intellidoc-backend.onrender.com`
- Backend API documentation: `https://tannu-intellidoc-backend.onrender.com/docs`

When Render asks for `LLM_API_KEY`, enter a valid Groq API key. Do not commit it to the repository. Keep the Blueprint service names unchanged so that the frontend API URL and backend CORS origin continue to match. If you rename either service, update `frontend/config.js` and the backend's `CORS_ALLOWED_ORIGINS` environment variable.

The free web plan has 512 MB of memory and sleeps after inactivity, so the first request after a sleep can be slow and high-resolution OCR may be constrained. The free Postgres database also expires after 30 days. For sustained use, select a paid web plan with at least 2 GB RAM and a paid database in the Render dashboard.

## Run with Docker

```bash
docker build -t intelligent-document-api .
docker run --rm -p 10000:10000 \
  -e LLM_API_KEY=your_groq_api_key \
  -e DATABASE_URL=sqlite:////tmp/intellidoc.sqlite3 \
  intelligent-document-api
```

The container starts only the backend. Open `http://localhost:10000/docs` for its API documentation.

## Run locally without Docker

Install Python 3.12 and Tesseract OCR, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --port 8000
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. In another terminal, serve the frontend:

```bash
python -m http.server 8080 --directory frontend
```

Open `http://localhost:8080`. The frontend automatically uses `http://localhost:8000` when served locally and the separate Render backend when deployed. Configuration defaults and supported backend variables are listed in `.env.example`.

## API endpoints

- `GET /api/v1/health` - service health
- `POST /api/v1/documents/process` - upload and process a PDF
- `GET /api/v1/documents` - list processed documents
- `GET /api/v1/documents/{document_name}` - retrieve one result
