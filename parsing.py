import requests
from bs4 import BeautifulSoup

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

response = requests.get("https://ru.wikipedia.org/wiki/ChatGPT")

if response is not None:
    html = BeautifulSoup(response.text, 'html.parser')

    title = html.select("#firstHeading")[0].text
    paragraphs = html.select("p")
    for para in paragraphs[:2]:
        print (para.text)