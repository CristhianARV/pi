techno : chromaDB, milvus et Qdrant à voir pour le vector store 

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
```

MOde console : /clear et /quit

Mode console ou gui :
```bash
uv run main.py Mode
```
remplacer le Mode avec soit console ou gui pour interface console ou graphique