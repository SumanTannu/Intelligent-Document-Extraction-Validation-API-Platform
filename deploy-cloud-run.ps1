param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    [string]$Region = "asia-south1",
    [string]$Repository = "intellidoc",
    [string]$BackendService = "intellidoc-backend",
    [string]$FrontendService = "intellidoc-frontend",
    [string]$LlmSecret = "intellidoc-llm-api-key",
    [string]$DatabaseSecret = "intellidoc-database-url"
)

$ErrorActionPreference = "Stop"
$imageTag = (git rev-parse --short HEAD).Trim()
if (-not $imageTag) { $imageTag = Get-Date -Format "yyyyMMddHHmmss" }
$registry = "$Region-docker.pkg.dev/$ProjectId/$Repository"

gcloud config set project $ProjectId
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com secretmanager.googleapis.com

$repositoryExists = gcloud artifacts repositories describe $Repository --location $Region --format "value(name)" 2>$null
if (-not $repositoryExists) {
    gcloud artifacts repositories create $Repository --repository-format docker --location $Region
}

gcloud builds submit . --config cloudbuild.yaml --substitutions "_REGION=$Region,_REPOSITORY=$Repository,COMMIT_SHA=$imageTag"
if ($LASTEXITCODE -ne 0) { throw "Cloud Build failed." }

gcloud run deploy $BackendService `
    --image "$registry/backend:$imageTag" `
    --region $Region `
    --no-invoker-iam-check `
    --port 8080 `
    --memory 2Gi `
    --cpu 1 `
    --timeout 900 `
    --concurrency 4 `
    --set-env-vars "LLM_PROVIDER=groq,LLM_MODEL=qwen/qwen3.8-27b,LLM_MAX_COMPLETION_TOKENS=4096,OCR_MULTI_PASS=true" `
    --set-secrets "LLM_API_KEY=${LlmSecret}:latest,DATABASE_URL=${DatabaseSecret}:latest"
if ($LASTEXITCODE -ne 0) { throw "Backend deployment failed." }

$backendUrl = (gcloud run services describe $BackendService --region $Region --format "value(status.url)").Trim()
if (-not $backendUrl) { throw "Could not determine the backend URL." }

gcloud run deploy $FrontendService `
    --image "$registry/frontend:$imageTag" `
    --region $Region `
    --no-invoker-iam-check `
    --port 8080 `
    --memory 256Mi `
    --set-env-vars "API_BASE_URL=$backendUrl"
if ($LASTEXITCODE -ne 0) { throw "Frontend deployment failed." }

$frontendUrl = (gcloud run services describe $FrontendService --region $Region --format "value(status.url)").Trim()
if (-not $frontendUrl) { throw "Could not determine the frontend URL." }

gcloud run services update $BackendService --region $Region --update-env-vars "CORS_ALLOWED_ORIGINS=$frontendUrl"
if ($LASTEXITCODE -ne 0) { throw "Backend CORS update failed." }

Write-Host "Frontend: $frontendUrl"
Write-Host "Backend:  $backendUrl"
Write-Host "API docs: $backendUrl/docs"
