from fastapi import FastAPI
from engine import tokenizer, model, qdrant_client, average_pool
from qdrant_client.http import models
import torch.nn.functional as F
import uvicorn
from keys import COLLECTION_NAME
import requests

app = FastAPI()

@app.get('/get_relevant_sentences_by_query')
def get_relevant_sentences_by_query(query: str, sentences_number: int = 10):
    """Получает строку-запрос и возвращает k наиболее релевантных предложений из базы данных"""
    batch_dict = tokenizer([query,], max_length=512, padding=True, 
                                truncation=True, return_tensors='pt')
            
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    embedding = F.normalize(embeddings, p=2, dim=1)[0]

    closest_vectors = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        search_params=models.SearchParams(hnsw_ef=128, exact=True),
        query_vector=embedding,
        limit=sentences_number,
        with_vectors=False,
        with_payload=True
    )

    return closest_vectors


if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0')