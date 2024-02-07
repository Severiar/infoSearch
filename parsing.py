import requests
from bs4 import BeautifulSoup
import re
from razdel import sentenize
import pandas as pd
import time
from fake_useragent import UserAgent
import random

def get_clear_string(string: str) -> str:
	removed_references = re.sub(r"\[[0-9]+\]", "", string)
	removed_nobreak_spaces = re.sub("\xa0", "\x20", removed_references)
	return removed_nobreak_spaces


def get_wiki_sentences_dataframe(articles_list: list[str]):
	sentences = []
	titles = []
	urls = []

	for article in articles_list:
		response = requests.get(f"https://ru.wikipedia.org/wiki/{article}", headers={'User-Agent': UserAgent().chrome})

		if response is not None:
			html = BeautifulSoup(response.text, 'html.parser')

			title = html.select("#firstHeading")[0].text
			paragraphs = html.select("p")
			for para in paragraphs:
				for sentence in list(sentenize(get_clear_string(para.text))):
					if len(sentence.text) < 30:
						continue
					sentences.append(sentence.text)
					titles.append(title)
					urls.append(f"https://ru.wikipedia.org/wiki/{article}")
		
		time.sleep(10 + 3 * random.random())

	return pd.DataFrame(data={'sentence': sentences, 'title': titles, 'url': urls})
