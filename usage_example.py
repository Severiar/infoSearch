import requests
import json

queries_list = [
    "ChatGPT является одним из авторов сценария эпизода мультсериала «Южный Парк».",
    "Площадь японского православного храма составляет лишь 16 м².",
    "Предок турецкого поэта был участником польского восстания",
    "Первой из женщин чашу олимпийского огня зажгла мексиканская бегунья.",
    "Самый большой стадион для игры в крикет вмещает 132 000 зрителей."

]
response = []
for query in queries_list:
    response.append(
        {
            "query": query,
            "relevant_sentences": requests.get('http://localhost:8000/get_relevant_sentences_by_query',
                                                params={
                                                    "query": query, 
                                                    "sentences_number": 10}).json()
        }
    )
out_file = open("usage_result_example.json", "w", encoding="utf-8") 

json.dump(response, out_file, indent = 4, ensure_ascii=False) 