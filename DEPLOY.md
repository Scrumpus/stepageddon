# Deploying Stepageddon

Single-VPS, Docker Compose, anonymous public access. Half-page runbook —
not exhaustive.

## Prerequisites on the host

- Docker + Docker Compose v2
- A domain pointing at the host with TLS terminated by something in front
  (Caddy / nginx-proxy / Cloudflare Tunnel). The compose stack publishes
  ports 80 (frontend) and 8000 (backend) on the host; the reverse proxy
  decides what's reachable from the internet.
- The trained model checkpoint copied to `backend/ml/checkpoints/`.
  The compose file mounts that directory read-only into the backend
  container, so swapping models doesn't require a rebuild.

## Environment

Copy the examples and fill them in:

```
cp backend/.env.example backend/.env       # backend runtime config
cp frontend/.env.example frontend/.env     # frontend build-time config
```

Also create a root-level `.env` for the compose substitutions
(`docker compose` reads this automatically):

```
# .env at repo root — used by docker-compose.yml
ENV=production

POSTGRES_USER=stepageddon
POSTGRES_PASSWORD=<long random string>
POSTGRES_DB=stepageddon

# Production frontend origin(s), comma-separated. No trailing slashes.
CORS_ORIGINS=https://stepageddon.example.com

# Used by the frontend build (Vite inlines it into the bundle).
VITE_API_URL=https://api.stepageddon.example.com

SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...

# If using S3 / R2 instead of local disk:
# STORAGE_BACKEND=s3
# S3_BUCKET=...
# S3_ENDPOINT_URL=...
# S3_ACCESS_KEY_ID=...
# S3_SECRET_ACCESS_KEY=...
```

The compose file fails closed if `POSTGRES_PASSWORD`, `CORS_ORIGINS`, or
`VITE_API_URL` are unset — that's intentional.

## First deploy

```
docker compose build
docker compose up -d
docker compose ps    # all three services should be (healthy) within ~60s
```

Smoke checks:

```
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:80/
```

## Updating

```
git pull
docker compose build
docker compose up -d
```

## Rolling back

```
docker compose down
git checkout <previous-known-good-tag-or-sha>
docker compose up -d --build
```

## Operational notes

- Per-IP rate limits: 5 requests/minute and 30/hour on the generation
  endpoints. Behind a reverse proxy, ensure the proxy forwards the real
  client IP via `X-Forwarded-For` — the backend trusts it
  (`--proxy-headers`). If you don't, every request appears to come from
  the proxy and limits become meaningless.
- Audio storage and the Postgres volume are not backed up automatically.
  Snapshot the `pgdata` Docker volume and the `backend/audio_storage`
  directory on whatever cadence you want.
- The OpenAPI docs (`/docs`, `/redoc`) are hidden when `ENV=production`.
- Generation can take 5–30s of CPU per song. Two uvicorn workers is the
  default — bump `--workers` in `backend/Dockerfile` if you have spare cores.
