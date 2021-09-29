import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re
paragraph =  """We are an emerging technologies firm that enables 
Hyperautomation through our proprietary platform, enterprise integrations,
and ISV partnerships. We chose the name Koi as it means "Any" (for any data format, unstructured or structured) in Hindi as well as accurately represents the underlying technology i.e. 
K(C)ognitive Optical Imaging.Our founders are alums of XPO Logistics, EY, PwC, Mu Sigma, Oracle, Blue Yonder/JDA/i2, and Capgemini. We offer deep expertise in Artificial Intelligence, EDGE Computing, Image Processing, Algorithms, Data Science, Logistics, Trade, Transportation, Maritime, and Supply Chain.

Our platform capabilities are backed by multiple patents.KoiReader Technologies is a GLOBAL AWARD WINNING FreightTech 100 company with deep industry and operations footprint. Our core product is AIoT-powered Hyperautomation Platform that includes Document Digitization and Machine Vision."""

# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

model = Word2Vec(sentences, min_count=1)

words=model.wv.key_to_index
