# Workflow

1. L'utilisateur choisit le mode: texte ou audio
2. Si audio: enregistrer la voix utilisateur
3. Afficher la transcription STT
4. Detecter l'emotion depuis la voix (modele entraine sur dataset audio)
5. Generer une reponse therapeutique adaptee a l'emotion
6. Sauvegarder la sortie de reponse dans `data/outputs`
