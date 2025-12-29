import nltk
from typing import List
from razdel import sentenize, tokenize

nf ='test.txt'
with open(nf, 'r', encoding="utf-8") as reader: #utf-8 т.к. у меня мак
    data = reader.read()

print("======NATASHA=======")

def parse_sentence(text: str) -> List[str]:
    return [s.text for s in sentenize(text)]

def get_tokens(text: str) -> List[str]:
    return [t.text for t in tokenize(text)]

# Токенизация по предложениям natasha
i = 1
print("По предложениям:")
for s in parse_sentence(data):
    print(i, '  ', s)
    i += 1
# Токенизация по словам natasha
print("По словам:")
print(get_tokens(data))

print("=======NLTK========")

# Токенизация по предложениям nltk
sentences = nltk.sent_tokenize(data , language="russian")
i=1
print("По предложениям:")
for sentence in sentences:
    print(i, '  ',sentence)
    i+=1
# Токенизация по словам nltk
print("По словам:")
words = nltk.word_tokenize(data)
print(words)