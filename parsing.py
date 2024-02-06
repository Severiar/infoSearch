import requests
from bs4 import BeautifulSoup
import re
from razdel import sentenize
import pandas as pd


def get_clear_string(string: str) -> str:
	removed_references = re.sub(r"\[[0-9]\]", "", string)
	removed_nobreak_spaces = re.sub("\xa0", "\x20", removed_references)
	return removed_nobreak_spaces


def get_wiki_sentences_dataframe(articles_list: list[str]):
	sentences = []
	titles = []
	urls = []

	for article in articles_list[:1]:
		response = requests.get(f"https://ru.wikipedia.org/wiki/{article}")

		if response is not None:
			html = BeautifulSoup(response.text, 'html.parser')

			title = html.select("#firstHeading")[0].text
			paragraphs = html.select("p")
			for para in paragraphs:
				for sentence in list(sentenize(get_clear_string(para.text))):
					sentences.append(sentence.text)
					titles.append(title)
					urls.append(f"https://ru.wikipedia.org/wiki/{article}")
	
	return pd.DataFrame(data={'sentence': sentences, 'title': titles, 'url': urls})


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

print(get_wiki_sentences_dataframe(articles))