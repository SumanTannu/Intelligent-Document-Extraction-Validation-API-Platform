# Intelligent Document Extraction, Validation & API Platform

A FastAPI application that accepts financial PDFs, extracts text with native PDF parsing or multi-pass Tesseract OCR, uses Groq for grounded structured extraction, validates financial totals, and stores processed results.

## Deploy both services on Google Cloud Run

The repository includes separate backend and frontend containers, plus a PowerShell deployment script. The frontend container writes its API URL at startup, and the script updates backend CORS after Cloud Run assigns both service URLs.

Prerequisites: install and authenticate the Google Cloud CLI, select a billing-enabled project, and create two Secret Manager secrets. `DATABASE_URL` should point to durable PostgreSQL; SQLite on Cloud Run is ephemeral.

```powershell
gcloud auth login
gcloud auth application-default login

gcloud secrets create intellidoc-llm-api-key --data-file="path/to/llm-key.txt"
gcloud secrets create intellidoc-database-url --data-file="path/to/database-url.txt"

.\deploy-cloud-run.ps1 -ProjectId "YOUR_GOOGLE_CLOUD_PROJECT"
```

The deployer needs permission to use Cloud Build, Artifact Registry, Cloud Run, and Secret Manager. The Cloud Run runtime service account also needs `Secret Manager Secret Accessor` for both secrets. Re-running the script builds a new revision and updates both services. The defaults deploy to `asia-south1`; override it with `-Region` if needed.

The services use Cloud Run's invoker-IAM-check setting for public access. This also works in organizations whose domain-restricted-sharing policy rejects an `allUsers` IAM binding.

For Cloud SQL, attach the instance to the backend service and use a Unix-socket SQLAlchemy URL in the database secret, for example `postgresql+psycopg2://USER:PASSWORD@/DATABASE?host=/cloudsql/PROJECT:REGION:INSTANCE`:

```powershell
gcloud run services update intellidoc-backend --region asia-south1 --add-cloudsql-instances PROJECT:REGION:INSTANCE
```

## Live services

IntelliDoc is deployed as separate frontend and backend services on Google Cloud Run:

- Frontend: [tannu-intellidoc-frontend-6fgf2nqd6a-el.a.run.app](https://tannu-intellidoc-frontend-6fgf2nqd6a-el.a.run.app)
- Backend: [tannu-intellidoc-backend-6fgf2nqd6a-el.a.run.app](https://tannu-intellidoc-backend-6fgf2nqd6a-el.a.run.app)
- API documentation: [Swagger UI](https://tannu-intellidoc-backend-6fgf2nqd6a-el.a.run.app/docs)

The backend Cloud Run service includes FastAPI and Tesseract OCR. Configure production secrets such as `LLM_API_KEY` in Cloud Run, not in the repository. If the frontend or backend URL changes, update `frontend/config.js` and the backend `CORS_ALLOWED_ORIGINS` configuration to match.

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

Open `http://localhost:8080`. The frontend automatically uses `http://localhost:8000` when served locally and the separate Google Cloud Run backend when deployed. Configuration defaults and supported backend variables are listed in `.env.example`.

## API endpoints

- `GET /api/v1/health` - service health
- `POST /api/v1/documents/process` - upload and process a PDF
- `GET /api/v1/documents` - list processed documents
- `GET /api/v1/documents/{document_name}` - retrieve one result
