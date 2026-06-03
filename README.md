# Installation python

La première étape consite à installer les dépendances du projet. On vous recommande d'utiliser un gestionnaire des packages comme uv.

Installation avec uv depuis la racine du projet:
```bash
uv add -r requirements.txt
```
Il se chargera d'installer toutes les dépendances nécessaires pour faire fonctionner le projet avec la bonne version de python. On a utilisé python 3.11. Si vous utiliser uv, la version de python est déjà definie dans le fichier .python-version à la racine du projet. Vous pouvez la changer si vous souhaitez utiliser une autre version de python, mais on vous recommande d'utiliser la même version que celle utilisée pour le développement du projet pour éviter les problèmes de compatibilité.

# Récupérer les données de la base de données

Créer un dossier backup à la racine du projet et y placer le fichier qdrant_storage.tar.gz. Attention, vous devez avoir installé docker ou docker desktop si vous êtes sur windows ou mac pour pouvoir utiliser le backup de la base de données. Le fichier qdrant_storage.tar.gz contient les données de la base de données Qdrant utilisée dans le projet. Il est important de le placer dans le dossier backup à la racine du projet pour pouvoir l'utiliser avec docker compose. Sinon vous aurez une base de données vide et vous devrez indexer les documents à nouveau pour pouvoir utiliser l'interface RAG. Voici la structure du projet après avoir ajouté le fichier de backup de la base de données:

```bash
PI/
├── docker-compose.yml
└── backup/
    └── qdrant_storage.tar.gz
```


# Qdrant in local
Utiliser docker compose depuis racine projet (Même repertoire que docker-compose.yml)
```bash 
docker compose up -d
```

# Lancer le projet
Avant de lancer le projet, vous devez vous connecter avec le vpn de l'he-arc pour accéder aux modèles  LLMs.

```bash
uv run main.py
```
Interface RAG disponible à l'adresse local URL:  http://127.0.0.1:7860


# Pour indexer documents

Si nécessaire, vous pouvez indexer les documents à nouveau en utilisant le script src/index_docs.py. Ce script se charge d'indexer les documents dans la base de données Qdrant.
```bash
uv run src/index_docs.py
```

Attention, définir chemin des documents à indexer et nom de la collection dans le code avant de lancer le script.


# Info utiles 

ancienne version docker compose
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333" # Interface HTTP / Dashboard
      - "6334:6334" # Interface gRPC
    volumes:
      - qdrant_storage:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_storage:
    name: qdrant_storage
```

Créer un backup de la base de données Qdrant en utilisant le volume Docker
```bash
docker run --rm `
  -v pi_qdrant_storage:/data `
  -v "${PWD}/backup:/backup" `
  alpine `
  sh -c "rm -rf /data/* && tar xzf /backup/qdrant_storage.tar.gz -C /data"
```

# Repertoire Notebook
Le repertoire notebook contient des notebooks pour tester les différentes fonctionnalités du projet. Vous pouvez les utiliser pour comprendre comment fonctionne le projet et pour tester les différentes fonctionnalités. Les notebooks sont organisés par thème, vous pouvez les trouver dans le repertoire notebook. Ils sont pas forcément à jour avec la dernière version du projet, mais ils peuvent vous aider à comprendre comment fonctionne le projet et comment utiliser les différentes fonctionnalités. N'hésitez pas à les utiliser et à les modifier pour vos besoins.