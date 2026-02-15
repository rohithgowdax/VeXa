# Vexa (PyTorch Transformer)

A production-ready full-stack web application for German to English machine translation.

- **Backend**: FastAPI + PyTorch + SentencePiece
- **Frontend**: React (Vite) + Axios
- **Model**: Trained Transformer checkpoint loaded at startup

---

## Project Structure

```text
translator-app/
├── backend/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── app/
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── translate.py
│   │   ├── schemas.py
│   │   └── config.py
│   ├── models/
│   │   └── transformer.pth
│   ├── tokenizer/
│   │   └── tokenizer.model
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   └── components/
│   │       └── Translator.jsx
│   ├── index.html
│   └── package.json
└── README.md
```

---

## Features

- Transformer model loaded once on API startup
- SentencePiece tokenization + detokenization
- Greedy decoding for translation
- REST API endpoint: `POST /translate`
- Frontend with:
  - input autofocus
  - loading state
  - error handling
  - responsive UI

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

---

## Setup

### 1) Add your trained artifacts

Replace these placeholder files with your real trained assets:

- `backend/models/transformer.pth`
- `backend/tokenizer/tokenizer.model`

> The API will fail to start if these files are invalid or missing.

### 2) Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) update `backend/.env` hyperparameters to match your trained checkpoint.

### 3) Frontend setup

```bash
cd frontend
npm install
```

---

## Run the Application

### Start backend

From `backend/`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend API: http://localhost:8000

### Start backend with Docker

From `backend/`:

```bash
docker build -t translator-backend:latest .
docker run --rm -p 8000:8000 translator-backend:latest
```

Backend API: http://localhost:8000

### Start frontend

From `frontend/`:

```bash
npm run dev
```

Frontend app: http://localhost:5173

---

## API Usage

### Endpoint

`POST /translate`

### Request

```json
{
  "text": "Das Wetter ist heute schön."
}
```

### Response

```json
{
  "translation": "The weather is nice today."
}
```

### cURL example

```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Das Wetter ist heute schön."}'
```

---

## Notes for Production

- Run FastAPI behind a reverse proxy (Nginx/Caddy).
- Enable HTTPS in production.
- Restrict CORS origins in `app/main.py`.
- Consider request logging, rate limiting, and auth for public deployments.
- Use a GPU-enabled environment for low-latency inference.
