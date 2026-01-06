# Novialoom Core Service

Ce dépôt contient le service Core de Novialoom, qui fournit une infrastructure LLM (Large Language Model) pour la génération de texte.

## 📂 Structure du Dépôt

*   **`services/core-service/`** : Service Core - Infrastructure LLM avec support Google Gemini et OpenAI
*   **`shared/`** : Code partagé (Modèles Pydantic, Auth, Configuration, Health checks, Logging)

## 🚀 Core Service

Le Core Service fournit une interface unifiée pour l'utilisation de différents providers LLM :
- **Google Gemini** (modèles 2.5) avec Google Search et Google Maps grounding
- **OpenAI** (GPT-4, GPT-3.5-turbo)
- **Embeddings** : Service d'embeddings pour la vectorisation de texte

Voir [services/core-service/README.md](services/core-service/README.md) pour plus de détails.

## ⚡ Architecture Serverless

Le service peut être packagé dans un container Docker et exposé via AWS Lambda grâce à **Mangum**.

### Comment ajouter Mangum au service ?
1.  Ajouter `mangum` dans le `requirements.txt`.
2.  Dans le `main.py` :
    ```python
    from mangum import Mangum
    from fastapi import FastAPI
    app = FastAPI()
    handler = Mangum(app, lifespan="off")
    ```

## 🛠️ Développement Local

Voir la documentation dans `services/core-service/README.md` pour les instructions de développement local.

---
*Dernière mise à jour : Janvier 2026*

