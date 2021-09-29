import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph =  """We are an emerging technologies firm that enables 
Hyperautomation through our proprietary platform, enterprise integrations,
and ISV partnerships. We chose the name Koi as it means "Any" (for any data format, unstructured or structured) in Hindi as well as accurately represents the underlying technology i.e. 
K(C)ognitive Optical Imaging.Our founders are alums of XPO Logistics, EY, PwC, Mu Sigma, Oracle, Blue Yonder/JDA/i2, and Capgemini. We offer deep expertise in Artificial Intelligence, EDGE Computing, Image Processing, Algorithms, Data Science, Logistics, Trade, Transportation, Maritime, and Supply Chain.

Our platform capabilities are backed by multiple patents.KoiReader Technologies is a GLOBAL AWARD WINNING FreightTech 100 company with deep industry and operations footprint. Our core product is AIoT-powered Hyperautomation Platform that includes Document Digitization and Machine Vision."""
               
               
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   