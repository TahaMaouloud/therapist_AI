# Therapist IA

Projet de fin d'etude: assistant therapeutique IA capable de:
- Ecouter l'utilisateur (audio)
- Transcrire la parole en texte (STT)
- Generer une reponse therapeutique textuelle (LLM)
- Produire une reponse vocale (TTS)

## Arborescence

- data/: donnees brutes, transcriptions et sorties
- src/: code source du pipeline
- docs/: architecture et workflow
- tests/: tests unitaires et integration
- models/: artefacts/modeles locaux

## Pipeline cible

1. Capture audio utilisateur
2. Transcription STT
3. Generation de reponse textuelle
4. Synthese vocale TTS
5. Sauvegarde des sorties (data/outputs)

## Entrainement emotion vocale (multi-datasets)

Par defaut, l'entrainement utilise:
- `data/raw_audio/archive` (RAVDESS)
- `data/AudioWAV` (2e dataset)

Commande:
`.\scripts\run_audio_training.ps1`

Resultats:
- `models/emotion_audio_model.joblib`
- `models/emotion_audio_metrics.json`

Le script compare les modeles demandes et selectionne automatiquement le meilleur
selon le `macro F1` de validation.

## Lancer l'application (console)

Commande:
`.\scripts\run_pipeline.ps1`

Au lancement, l'utilisateur choisit:
- mode texte
- mode audio (STT + detection emotion vocale)

## Frontend React

Le frontend est dans `frontend/`.

## Backend Spring Boot

Le backend Spring Boot est dans `backend/`.

Il gere:
- auth (`/auth/register`, `/auth/verify-email`, `/auth/login`, `/auth/me`)
- endpoints therapie (`/session/text`, `/session/text-auth`, `/session/audio-upload`)

Ce backend appelle ensuite le service Python IA pour:
- STT local
- detection emotion vocale
- reponse therapeutique

Service Python IA (toujours necessaire):
`uvicorn src.api.server:app --reload --port 8000`

Module reponse therapeutique local (LLaMA + fallback):
- Le moteur utilise un modele local LLaMA via `llama_cpp` (format GGUF).
- Si le modele local est indisponible, le fallback heuristique reste actif automatiquement.
- Historique en memoire par session (`session_id`) + ton adapte a l'emotion detectee.
- Regles de securite actives: blocage des reponses dangereuses et reponse de crise.

Variables utiles:
- `THERAPIST_LLM_ENABLED` (`true`/`false`, defaut `true`)
- `THERAPIST_LLM_PROVIDER` (`llama_cpp` ou `rule_based`)
- `THERAPIST_LLAMA_MODEL_PATH` (ex: `models/llama/llama-3.1-8b-instruct.Q4_K_M.gguf`)
- `THERAPIST_LLAMA_RUNTIME_PATH` (runtime local optionnel pour chargement direct, ex: `vendor/llama_cpp_runtime`)
- `THERAPIST_LLAMA_HELPER_PYTHON` (runtime Python helper, ex: `.tmp/python312_runtime/python.exe`)
- `THERAPIST_LLAMA_HELPER_RUNTIME_PATH` (packages `llama_cpp` du helper, ex: `vendor/llama_cpp_runtime312`)
- `THERAPIST_LLAMA_PREFER_HELPER` (`true` pour preferer le helper Windows stable)
- `THERAPIST_LLAMA_N_CTX` / `THERAPIST_LLAMA_N_THREADS` (optimisation CPU)
- `THERAPIST_HISTORY_MAX_TURNS` (defaut `8`)
- `THERAPIST_LLM_MAX_TOKENS` (defaut `24`)
- `THERAPIST_LLM_HISTORY_MAX_TURNS` (defaut `3`)
- `THERAPIST_LLM_TEMPERATURE` (defaut `0.6`)
- `THERAPIST_LLM_TOP_P` (defaut `0.9`)

Qualite STT:
- `STT_MODEL` (faster-whisper local, ex: `small`, `medium`, `large-v3`)
- `STT_LANGUAGE` (`en` par defaut)
- `STT_DEVICE` (`cpu` ou `cuda`)
- `STT_COMPUTE_TYPE` (ex: `int8`, `float16`)
- `STT_INPUT_DEVICE` (index micro, ou `auto`)
- `STT_MIN_RMS` (seuil anti-silence, ex: `0.01`)

Backend Spring:
1. `cd backend`
2. `mvn spring-boot:run`
ou depuis la racine:
`.\scripts\run_backend.ps1`

Le backend tourne sur `http://127.0.0.1:8080`.

Frontend:
1. `cd frontend`
2. `npm install`
3. `npm run dev`

Le frontend appelle ces routes:
- `POST /auth/register`, `POST /auth/verify-email`, `POST /auth/login`, `GET /auth/me` (Spring)
- `POST /session/text` ou `POST /session/text-auth` (Spring -> Python IA)
- `POST /session/audio-upload` (Spring -> Python IA)   

Liaison au modele:
- `session/audio-upload` envoie le fichier au backend
- backend fait STT local (`src/stt/transcriber.py`)
- backend detecte emotion voix (`src/nlp/emotion_audio.py`)
- backend genere la reponse therapeutique (`src/nlp/therapist_agent.py`)
- les reponses audio FastAPI incluent aussi la double prediction emotion voix:
  `audio_emotion_top2`, `audio_emotion_primary`, `audio_emotion_secondary`

## Deploiement simple (debutant) avec Docker

Objectif: lancer toute l'app avec une seule stack:
- `web` (Nginx + frontend React)
- `backend` (Spring Boot)
- `python-service` (FastAPI IA)
- `postgres` (base de donnees)

### 1) Preparer les variables

Depuis la racine du projet:

```bash
cp .env.prod.example .env
```

Sous PowerShell (Windows):

```powershell
Copy-Item .env.prod.example .env
```

Puis ouvre `.env` et change au minimum:
- `POSTGRES_PASSWORD`
- `DB_PASSWORD`
- `CORS_ALLOWED_ORIGINS` (mets ton domaine frontend en production, ex: `https://ton-domaine.com`)
- `MAIL_*` si tu veux envoyer des codes email reels

### 2) Build et lancement

```bash
docker compose build 
docker compose up -d 
```

### 3) Verifier

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f python-service
```

Ton application sera accessible sur:
- `http://IP_DU_SERVEUR` (frontend)
- API via Nginx: `http://IP_DU_SERVEUR/api/...`

### 4) Arreter

```bash
docker compose down
```

### 5) Mise a jour (nouvelle version)

```bash
git pull
docker compose build
docker compose up -d
```
