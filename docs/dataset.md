# Data

- data/raw_audio: enregistrements originaux
- data/transcripts: texte transcrit
- data/processed: donnees pretraitees
- data/outputs: reponses finales texte/audio
- data/processed/emotions.csv: dataset d'entrainement emotion (conseille)
- data/processed/emotions_wav2vec2_en.csv: dataset texte anglais aligne avec labels wav2vec2 (`angry`, `fearful`, `happy`, `neutral`, `sad`)
- data/raw_audio/archive: base audio emotion (RAVDESS .wav)

Format recommande pour l'entrainement emotion:
- Colonnes minimales: `text`, `emotion`
- Exemple CSV:
  text,emotion
  "I feel very sad and tired",sad
  "I feel calm and normal today",neutral
  "I am happy with my progress",happy

Format pour l'entrainement emotion vocale:
- Fichiers `.wav` nommes comme RAVDESS (ex: `03-01-05-01-01-01-01.wav`)
- Emotion extraite depuis le 3e segment du nom de fichier
