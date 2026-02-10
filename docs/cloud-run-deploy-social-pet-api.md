# Cloud Run Deploy: Social Pet API

This deploy path runs the Fastify service (`services/social-pet-api`) on Google Cloud Run.

## Chosen defaults

- Project: `social-ai-pet`
- Project number: `746499514974`
- Region: `us-central1`
- Service: `social-pet-api-dev`
- Public endpoint: enabled
- Initial CORS allowlist: `http://localhost:5173`

## 0) Prerequisites

- Install and auth gcloud:

```bash
gcloud auth login
gcloud config set project social-ai-pet
```

- Enable required APIs:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

## 1) Create/update API key secrets

Use Secret Manager; do not place provider keys in source or `.env` committed files.

```bash
# OpenAI
gcloud secrets describe OPENAI_API_KEY >/dev/null 2>&1 || gcloud secrets create OPENAI_API_KEY --replication-policy=automatic
read -s OPENAI_API_KEY && printf "%s" "$OPENAI_API_KEY" | gcloud secrets versions add OPENAI_API_KEY --data-file=-

# Anthropic
gcloud secrets describe ANTHROPIC_API_KEY >/dev/null 2>&1 || gcloud secrets create ANTHROPIC_API_KEY --replication-policy=automatic
read -s ANTHROPIC_API_KEY && printf "%s" "$ANTHROPIC_API_KEY" | gcloud secrets versions add ANTHROPIC_API_KEY --data-file=-
```

## 2) Grant Cloud Run runtime access to secrets

Using default compute runtime service account:

```bash
gcloud secrets add-iam-policy-binding OPENAI_API_KEY --member="serviceAccount:746499514974-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding ANTHROPIC_API_KEY --member="serviceAccount:746499514974-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

## 3) Build and deploy container

From repo root:

```bash
gcloud builds submit --project=social-ai-pet --tag us-central1-docker.pkg.dev/social-ai-pet/cloud-run-source-deploy/social-pet-api-dev -f services/social-pet-api/Dockerfile .
```

```bash
gcloud run deploy social-pet-api-dev \
  --project=social-ai-pet \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --image=us-central1-docker.pkg.dev/social-ai-pet/cloud-run-source-deploy/social-pet-api-dev \
  --set-env-vars=LLM_PROVIDER=openai,OPENAI_MODEL=gpt-4.1-mini,ANTHROPIC_MODEL=claude-3-5-haiku-latest,LLM_TIMEOUT_MS=1200,LLM_HISTORY_TURNS=8,PERSISTENCE_MODE=memory,EVENT_LOG_MAX=200,CORS_ORIGINS=http://localhost:5173 \
  --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:latest,ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest \
  --port=8080
```

## 4) Smoke test deployed API

```bash
SERVICE_URL=$(gcloud run services describe social-pet-api-dev --project=social-ai-pet --region=us-central1 --format='value(status.url)')
curl "$SERVICE_URL/healthz"
curl -X POST "$SERVICE_URL/session/start" -H "content-type: application/json" -d '{}'
```

## 5) Point web app to Cloud Run URL

Set `/Users/jeffreythomas/Documents/social-ai/apps/social-pet-web/.env`:

```bash
VITE_API_BASE_URL=<SERVICE_URL_FROM_STEP_4>
```

Then run:

```bash
yarn dev:web
```
