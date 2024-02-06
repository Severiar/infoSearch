from parsing import get_wiki_sentences_dataframe
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from keys import QDRANT_URL, QDRANT_API_KEY
import numpy as np


qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
)

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')


def create_new_collection(collection_name: str) -> None:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024, 
            distance=models.Distance.COSINE
        ),
    )


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def upsert_wiki_database(articles_names: list[str], collection_name: str, 
                         start_id: int, create_collection: bool = False) -> None:
    if create_collection:
        create_new_collection(collection_name)
    
    for article in articles_names:
        article_subframe = get_wiki_sentences_dataframe([article])
        print(article_subframe)
        for index, batch in enumerate(np.array_split(article_subframe, len(article_subframe))):

            batch_dict = tokenizer(list(batch['sentence']), max_length=512, padding=True, 
                                truncation=True, return_tensors='pt')
            
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            qdrant_client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=[start_id + i for i in range(batch.shape[0])],
                    payloads=[
                        {
                            "text": row["sentence"],
                            "title": row["title"],
                            "url": row["url"],
                        }
                        for _, row in batch.iterrows()
                    ],
                    vectors=[v.tolist() for v in embeddings],
                ),
            )
            start_id += batch.shape[0]

            print(f"Batch {index}: Successfully added {len(embeddings)} vectors to collection '{collection_name}'.")


articles = [
		'ChatGPT',
	    'Виртуальный_собеседник',
		'Генеративный_искусственный_интеллект',
		'Глубокое_обучение_(Южный_Парк)',
		'Южный_Парк',
		'Киноискусство',
		'Православная_церковь_Кумамото',
		'Японская_православная_церковь',
		'Автономная_церковь',
		'Божецкий,_Константин',
		'Назым_Хикмет',
		'Великопольское_восстание_(1848)',
		'Революция_1848—1849_годов_в_Германии',
		'Олимпийский_огонь',
		'Олимпийские_игры',
		'Басилио,_Энрикета',
		'Летние_Олимпийские_игры_1968',
		'Стадион_имени_Нарендры_Моди',
		'Крикет'
	]

ans = qdrant_client.search(
    collection_name="wiki collection",
    query_vector=[1 for i in range(1024)],
    limit=3,
)
print(ans)
# qdrant_client.delete(
#    collection_name="wiki collection",
#    points_selector=models.PointIdsList(
#        points=[i for i in range(279)],
#    ),
# )