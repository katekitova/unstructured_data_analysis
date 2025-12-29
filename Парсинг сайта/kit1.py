import requests
from bs4 import BeautifulSoup
import nltk

url = "https://al.cs.msu.ru/about.html"
response = requests.get(url)
response.encoding = "utf-8"
html = response.text
soup = BeautifulSoup(html, "html.parser")
paragraphs = soup.find_all("p")
data = " ".join([p.get_text(strip=True) for p in paragraphs])
sentences = nltk.sent_tokenize(data , language="russian")
i=1
for sentence in sentences:
    print(i,')  ',sentence)
    i+=1
print("Кол-предложений: ",len(sentences))

"""
Два раза будет выводиться "Заведующий кафедрой - профессор Н.В.Лукашевич" 
Т.к. в строке два раза <p><p>: <p><p>Заведующий кафедрой - профессор Н.В.Лукашевич.</p></p>
Так что по факту всего 13 приложений, а не 14
"""

