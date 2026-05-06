# Guide Detaille Du Projet Therapist AI

Version: 2026-02-15
Public cible: presentation technique pour encadrement

## 1) Vue globale du systeme

Ce projet est une application complete composee de 4 blocs:

1. Frontend React (port 5173) pour l'interface utilisateur.
2. Backend Spring Boot (port 8080) pour auth, orchestration et persistence.
3. Base PostgreSQL pour stocker utilisateurs et historique des sessions.
4. Service Python FastAPI (port 8000) pour STT + detection emotion + generation de reponse.

Flux principal:

1. L'utilisateur interagit depuis React.
2. React appelle Spring Boot.
3. Spring lit/ecrit PostgreSQL pour auth et historique.
4. Spring appelle FastAPI Python pour l'intelligence audio/texte.
5. FastAPI renvoie les resultats a Spring.
6. Spring renvoie la reponse finale a React.

## 2) Pipeline technique demande (entrainement -> STT -> TTS)

### 2.1 Entrainement du modele d'emotion texte

Objectif:
- Apprendre un classifieur emotion a partir d'un dataset texte (`text`, `emotion`).

Fichiers impliques:
- `scripts/run_training.ps1`
- `src/nlp/train_emotion_model.py`
- `models/emotion_model.joblib` (sortie generee)
- `models/emotion_metrics.json` (sortie generee)

Etapes:
1. Charger dataset CSV/JSON.
2. Nettoyer et verifier colonnes.
3. Vectoriser avec TF-IDF.
4. Entrainer LogisticRegression.
5. Evaluer (accuracy + classification report).
6. Sauvegarder modele + metriques.

### 2.2 Entrainement du modele d'emotion audio

Objectif:
- Predire l'emotion vocale depuis des fichiers audio (RAVDESS style).

Fichiers impliques:
- `scripts/run_audio_training.ps1`
- `scripts/train_model.ps1`
- `src/nlp/train_emotion_audio_model.py`
- `src/nlp/emotion_audio.py`
- `models/emotion_audio_model.joblib` (sortie generee)
- `models/emotion_audio_metrics.json` (sortie generee)

Etapes:
1. Scanner `data/raw_audio/archive` pour trouver les `.wav`.
2. Extraire label emotion depuis le nom du fichier RAVDESS.
3. Extraire features audio (MFCC, delta, chroma, mel, prosodie, spectre, etc.).
4. Entrainer et comparer plusieurs modeles (RandomForest, ExtraTrees).
5. Garder le meilleur sur validation (macro F1).
6. Evaluer sur test.
7. Sauvegarder modele + label encoder + metriques.

### 2.3 STT (Speech-To-Text)

Une implementation est active:

1. STT local:
- `src/stt/transcriber.py`
- Utilise `faster-whisper` en local CPU.
- Supporte transcription fichier + mode live micro.

`src/stt/transcribe_speech.py` est conserve uniquement comme alias vers `transcriber.py`.

### 2.4 TTS (Text-To-Speech)

Etat actuel:
- `src/tts/synthesizer.py` et `src/tts/synthesize_voice.py` sont des placeholders.
- Au lieu de generer un audio, ils ecrivent la reponse texte dans `data/outputs/reply.txt`.

Conclusion:
- Le pipeline TTS est prepare structurellement, mais pas encore branche a un moteur vocal reel.

## 3) Comment React, Spring, PostgreSQL et FastAPI sont relies

### 3.1 Cote React

Fichier cle:
- `frontend/src/App.jsx`

Parametre cle:
- `API_BASE = "http://127.0.0.1:8080"`

Donc React parle uniquement a Spring.

### 3.2 Cote Spring (backend central)

Fichiers cles:
- `backend/src/main/resources/application.yml`
- `backend/src/main/java/com/therapist/backend/controller/AuthController.java`
- `backend/src/main/java/com/therapist/backend/controller/TherapyController.java`
- `backend/src/main/java/com/therapist/backend/therapy/PythonModelClient.java`

Roles:
1. Expose routes `/auth/*` et `/session/*` pour le frontend.
2. Sauvegarde les donnees en PostgreSQL via JPA.
3. Appelle FastAPI (`python.service.base-url`) pour inference IA.

### 3.3 Cote PostgreSQL

Configuration:
- `backend/src/main/resources/application.yml` (`spring.datasource.url`, user, password)

Entites JPA:
- `backend/src/main/java/com/therapist/backend/auth/UserEntity.java` -> table `users`
- `backend/src/main/java/com/therapist/backend/therapy/TherapyHistoryEntity.java` -> table `therapy_history`

Repositories:
- `backend/src/main/java/com/therapist/backend/auth/UserRepository.java`
- `backend/src/main/java/com/therapist/backend/therapy/TherapyHistoryRepository.java`

Si PostgreSQL est arrete:
- Le backend Spring ne peut pas fonctionner normalement pour auth et historique.

### 3.4 Cote FastAPI

Fichier cle:
- `src/api/server.py`

Endpoints utiles:
- `/session/text`
- `/session/audio-upload`
- `/health`

Spring appelle FastAPI via:
- `backend/src/main/java/com/therapist/backend/therapy/PythonModelClient.java`

### 3.5 Parcours de bout en bout (exemple audio)

1. User clique "Uploader audio" dans React.
2. React envoie `POST /session/audio-upload` vers Spring (8080).
3. Spring `TherapyController` recoit le fichier.
4. Spring transfere le fichier a FastAPI via `PythonModelClient`.
5. FastAPI transcrit audio + detecte emotion + genere reponse.
6. FastAPI renvoie JSON a Spring.
7. Spring sauvegarde l'historique en DB (si utilisateur authentifie).
8. Spring renvoie la reponse a React.
9. React affiche transcript, emotion, reponse.

## 4) Description fichier par fichier

### 4.1 Racine du projet

| Fichier | Role principal | Details |
|---|---|---|
| `.env.example` | Exemple de variables d'environnement Python | Montre `STT_MODEL`, `TTS_MODEL`, `LANG`. |
| `.gitignore` | Regles Git | Ignore environnements, builds frontend/backend, outputs data, modeles generes. |
| `README.md` | Documentation generale | Explique architecture, commandes de lancement, ports, routes principales. |
| `pyproject.toml` | Meta Python minimale | Nom du projet, version, description, version Python requise. |
| `requirements.txt` | Dependances Python | FastAPI, ML audio/texte, STT local, etc. |

### 4.2 Documentation fonctionnelle/technique

| Fichier | Role principal | Details |
|---|---|---|
| `docs/architecture.md` | Resume architecture | Decrit blocs Audio, STT, LLM, TTS, storage. |
| `docs/workflow.md` | Workflow metier | Decrit sequence utilisateur texte/audio et sortie. |
| `docs/dataset.md` | Format des donnees | Donne conventions dataset texte + audio (RAVDESS). |

### 4.3 Scripts d'execution

| Fichier | Role principal | Details |
|---|---|---|
| `scripts/run_all.ps1` | Demarrage global | Ouvre 3 shells: FastAPI (8000), Spring (8080), React (5173). |
| `scripts/run_backend.ps1` | Demarrage backend Spring | Injecte variables DB/SMTP puis lance `mvn spring-boot:run`. |
| `scripts/run_pipeline.ps1` | Lancement pipeline console Python | Lance `python -m src.main`. |
| `scripts/run_audio_training.ps1` | Entrainement emotion audio | Lance `src.nlp.train_emotion_audio_model`. |
| `scripts/run_training.ps1` | Entrainement emotion texte | Lance `src.nlp.train_emotion_model`. |
| `scripts/train_model.ps1` | Alias entrainement audio | Duplique `run_audio_training.ps1`. |

### 4.4 Frontend React

| Fichier | Role principal | Details |
|---|---|---|
| `frontend/package.json` | Config npm | Dependances React/Vite + scripts `dev`, `build`, `preview`. |
| `frontend/package-lock.json` | Verrouillage versions npm | Fichier auto-genere pour reproduire les installs. |
| `frontend/vite.config.js` | Config serveur dev | Fixe le port Vite sur 5173. |
| `frontend/index.html` | HTML racine SPA | Contient `<div id="root"></div>` pour monter React. |
| `frontend/src/main.jsx` | Point d'entree React | Monte le composant `App` et charge `styles.css`. |
| `frontend/src/App.jsx` | Logique UI complete | Auth (inscription/verification/login code), chat texte, voix, upload audio, profil, navigation ecrans, appels API backend. |
| `frontend/src/styles.css` | Style global UI | Theme, layout dashboard/chat/voice, boutons, responsive mobile. |

### 4.5 Service Python (pipeline IA + API)

| Fichier | Role principal | Details |
|---|---|---|
| `src/main.py` | Point d'entree console | Lance la session pipeline interactive. |
| `src/config.py` | Settings Python | Lit variables d'environnement (modele STT/TTS, langue). |
| `src/utils/logger.py` | Logger placeholder | Fonction vide `setup_logger()` a completer. |
| `src/core/pipeline.py` | Orchestration console | Gere mode texte/audio, STT, emotion, reponse et sortie TTS placeholder. |
| `src/core/run_pipeline.py` | Doublon orchestration | Version equivalente de `pipeline.py` (redondance technique). |
| `src/audio/record_audio.py` | Capture micro simple | Enregistre X secondes et sauve WAV dans `data/raw_audio`. |
| `src/audio/recorder.py` | Doublon capture audio | Meme contenu que `record_audio.py` (redondant). |
| `src/stt/transcriber.py` | STT principal local | Charge `faster-whisper`, transcrit fichier et mode live chunk par chunk. |
| `src/stt/transcribe_speech.py` | Alias STT local | Re-exporte les fonctions de `src/stt/transcriber.py`. |
| `src/tts/synthesizer.py` | TTS placeholder | Ecrit la reponse dans un fichier texte `data/outputs/reply.txt`. |
| `src/tts/synthesize_voice.py` | Doublon TTS placeholder | Meme role que `synthesizer.py`. |
| `src/nlp/therapist_agent.py` | Generation reponse therapeutique | Moteur local regle-based: validation + piste concrete + question selon emotion. |
| `src/nlp/emotion_text.py` | Detection emotion texte heuristique | Dictionnaires de mots-cles et normalisation du texte vers labels emotion. |
| `src/nlp/train_emotion_model.py` | Entrainement texte ML | TF-IDF + LogisticRegression + export modele/metriques. |
| `src/nlp/emotion_audio.py` | Inference emotion audio actuelle | Features enrichies, support multi-format audio, map labels, prediction + confiance. |
| `src/nlp/train_emotion_audio_model.py` | Entrainement audio actuel | Build dataset RAVDESS, selection de modele, export bundle + metrics. |
| `src/api/server.py` | API FastAPI principale | Auth locale JSON, sessions, endpoints texte/audio, upload audio, fusion emotion audio+texte. |
| `src/api/api_server.py` | API FastAPI simplifiee | Version plus courte centree session texte/audio sans auth complete. |
| `src/api/auth_store.py` | Stockage auth local JSON | Cree utilisateurs/sessions dans `data/processed/*.json` (fallback hors PostgreSQL). |

### 4.6 Backend Spring Boot

| Fichier | Role principal | Details |
|---|---|---|
| `backend/pom.xml` | Build Maven | Dependances Spring Web/JPA/Mail, PostgreSQL, validation, tests. |
| `backend/src/main/resources/application.yml` | Configuration runtime | Port 8080, connexion PostgreSQL, SMTP, limite upload, URL FastAPI. |
| `backend/src/main/java/com/therapist/backend/TherapistBackendApplication.java` | Main Spring Boot | Demarre l'application Java. |
| `backend/src/main/java/com/therapist/backend/config/CorsConfig.java` | CORS + fichiers statiques | Autorise origines localhost, expose `/files/profile_photos/**`. |
| `backend/src/main/java/com/therapist/backend/controller/HealthController.java` | Health endpoint | Route `/health` pour supervision. |
| `backend/src/main/java/com/therapist/backend/controller/AuthController.java` | API auth HTTP | Register, verify-email, login 2 etapes, me, upload/suppression photo. |
| `backend/src/main/java/com/therapist/backend/controller/TherapyController.java` | API sessions therapy | Routes `/session/text`, `/session/text-auth`, `/session/audio-upload`, `/session/history`. |
| `backend/src/main/java/com/therapist/backend/auth/AuthDtos.java` | DTO auth | Modeles request/response validates. |
| `backend/src/main/java/com/therapist/backend/auth/AuthEmailService.java` | Envoi mails OTP | Envoie code verification et code connexion via SMTP. |
| `backend/src/main/java/com/therapist/backend/auth/AuthService.java` | Logique metier auth | Hash password, creation user, verification code email/login, gestion sessions en memoire. |
| `backend/src/main/java/com/therapist/backend/auth/UserEntity.java` | Entite JPA user | Mapping table `users` (email, password hash, codes, photo, etc.). |
| `backend/src/main/java/com/therapist/backend/auth/UserProfile.java` | Modele legacy user | Classe profile non-JPA encore presente (probable heritage). |
| `backend/src/main/java/com/therapist/backend/auth/UserRepository.java` | Repository users | Queries `findByEmail`, `findByUsername`, `findByVerifyToken`. |
| `backend/src/main/java/com/therapist/backend/therapy/PythonModelClient.java` | Client HTTP vers FastAPI | Appels JSON texte + multipart audio sur service Python. |
| `backend/src/main/java/com/therapist/backend/therapy/TherapyDtos.java` | DTO therapy | Request texte avec validation. |
| `backend/src/main/java/com/therapist/backend/therapy/TherapyHistoryEntity.java` | Entite JPA historique | Mapping table `therapy_history` (texte/transcript/emotion/reponse/date). |
| `backend/src/main/java/com/therapist/backend/therapy/TherapyHistoryRepository.java` | Repository historique | Recherche des 100 dernieres sessions par utilisateur. |
| `backend/src/main/java/com/therapist/backend/therapy/TherapyHistoryService.java` | Service persistence sessions | Sauvegarde sessions texte/audio et lecture historique. |
| `backend/error.log` | Journal de lancement Maven/Spring | Capture warnings/infos/erreurs lors d'un run backend. |

### 4.7 Donnees et artefacts

| Fichier | Role principal | Details |
|---|---|---|
| `models/.gitkeep` | Maintenir dossier versionne | Permet de conserver le dossier `models/` meme vide dans Git. |
| `backend/data/outputs/profile_photos/df26bbf5-565d-45eb-add2-91b3a71421f5.png` | Exemple fichier runtime | Photo de profil uploadee via API auth. |
| `backend/data/outputs/profile_photos/cf570859-2f52-4e9e-8068-fd5dafc1e3a9.PNG` | Exemple fichier runtime | Photo de profil uploadee via API auth. |
| `backend/data/outputs/profile_photos/b66a0750-1d2d-4335-9416-ad6b6bdc5176.PNG` | Exemple fichier runtime | Photo de profil uploadee via API auth. |
| `backend/data/outputs/profile_photos/b03b7387-4f54-479c-8449-4c173bad25fa.png` | Exemple fichier runtime | Photo de profil uploadee via API auth. |
| `backend/data/outputs/profile_photos/91b45dc5-f796-4f06-9a2a-cbc7fe74dda9.PNG` | Exemple fichier runtime | Photo de profil uploadee via API auth. |
| `backend/data/outputs/profile_photos/2124904a-596a-4cc3-9658-cb9683fe934e.PNG` | Exemple fichier runtime | Photo de profil uploadee via API auth. |

### 4.8 Tests

| Fichier | Role principal | Details |
|---|---|---|
| `tests/test_pipeline.py` | Test placeholder | Test minimal `assert True` pour valider pipeline CI basique. |

## 5) Sequences metier detaillees

### 5.1 Inscription + verification email + connexion 2 etapes

1. React envoie formulaire inscription a `POST /auth/register` (Spring).
2. Spring valide et cree user dans PostgreSQL.
3. Spring envoie code de verification par email (SMTP).
4. User saisit code -> `POST /auth/verify-email`.
5. User saisit email/mot de passe -> `POST /auth/login` (envoi code de connexion).
6. User saisit code login -> `POST /auth/login/verify-code`.
7. Spring renvoie `access_token`.
8. React stocke token et appelle `/auth/me`.

### 5.2 Session texte

1. React envoie texte a `POST /session/text-auth` (Spring).
2. Spring appelle FastAPI `POST /session/text`.
3. FastAPI detecte emotion texte + genere reponse.
4. Spring sauvegarde historique en PostgreSQL.
5. React affiche emotion + reponse.

### 5.3 Session audio

1. React envoie audio a `POST /session/audio-upload` (Spring).
2. Spring transfere fichier vers FastAPI `/session/audio-upload`.
3. FastAPI:
   - Sauvegarde le fichier.
   - Lance STT local (`transcriber.py`).
   - Lance emotion audio (`emotion_audio.py`) + fallback emotion texte.
   - Genere reponse (`therapist_agent.py`).
4. Spring sauvegarde resultat dans `therapy_history`.
5. React affiche transcript/emotion/reponse.

## 6) Ordre de demarrage recommande

1. Demarrer PostgreSQL.
2. Demarrer FastAPI Python (`uvicorn src.api.server:app --reload --port 8000`).
3. Demarrer Spring (`scripts/run_backend.ps1` ou `mvn spring-boot:run`).
4. Demarrer React (`cd frontend && npm run dev`).
5. Ouvrir `http://localhost:5173`.

Script tout-en-un:
- `scripts/run_all.ps1` peut lancer les 3 services applicatifs (Python + Spring + React), mais PostgreSQL doit deja etre actif.

## 7) Points techniques importants pour soutenance

1. Architecture hybride:
   - Java Spring pour auth, securite et persistance.
   - Python FastAPI pour IA/STT/emotion.

2. Separation des responsabilites:
   - Frontend: UX.
   - Backend Spring: orchestration et DB.
   - FastAPI: intelligence et traitement audio.

3. Extensibilite:
   - TTS reel peut remplacer facilement le placeholder.
   - Emotion texte peut passer d'heuristique a modele appris.
   - Auth JSON local FastAPI existe pour tests, mais la prod actuelle passe par Spring + PostgreSQL.

4. Robustesse:
   - Detection emotion audio avec score de confiance et fallback emotion texte.
   - Validation DTO cote Spring.
   - Historisation des sessions utilisateur.
