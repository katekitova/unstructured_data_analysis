from natasha import NewsEmbedding, NewsMorphTagger, Doc, Segmenter, MorphVocab
from nltk.stem import SnowballStemmer
from pymorphy3 import MorphAnalyzer

print("======NATASHA=======")
# text = "Мама мыла раму"
text = "У мамы не было мыла"
# text = "Домашка по машинке кринжатина полная"
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph = MorphVocab()
doc = Doc(text)
doc.segment(segmenter)
doc.tag_morph(morph_tagger)
for token in doc.tokens:
    token.lemmatize(morph)
for token in doc.tokens:
    print(f"{token.text:>5} | lemma={token.lemma:<5} | pos={token.pos:<5} | feats={token.feats}")

print("\n=======PYMORPHY========")
morph = MorphAnalyzer()
tokens = text.split()
for word in tokens:
    parses = morph.parse(word)
    print(f"\nСлово: {word}")
    for p in parses:
        print(p)

print("\n=======SNOWBALL========")
stemmer = SnowballStemmer("russian")
stems = [stemmer.stem(word) for word in tokens]
print(stems)