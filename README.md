# Récupérer les données de la base de données

Créer un dossier backup à la racine du projet et y placer le fichier qdrant_storage.tar.gz

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
```bash
uv run main.py
```
Interface RAG disponible à l'adresse local URL:  http://127.0.0.1:7860


# Pour indexer documents
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