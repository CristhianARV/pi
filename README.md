techno : chromaDB, milvus et Qdrant à voir pour le vector store 
BGE-M3

# Qdrant in local
Utiliser docker compose depuis racine projet (Même repertoire que docker-compose.yml)
```bash 
docker compose up -d
```


Ancienne version à la main. Pas besoin de le faire !!!!!!! 
```bash
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

#-----------------------
# Windows
docker volume create qdrant_storage
docker volume ls
docker volume inspect qdrant_storage
docker run --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
``

# Lancer le projet
```bash
uv run main.py
```
Systeme disponible à l'adresse local URL:  http://127.0.0.1:7860


# Pour indexer documents
```bash
uv run src/index_docs.py
```

Attention, définir chemin des documents à indexer et nom de la collection dans le code avant de lancer le script.