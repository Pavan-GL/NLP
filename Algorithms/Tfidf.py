import nltk

paragraph =  """We are an emerging technologies firm that enables 
Hyperautomation through our proprietary platform, enterprise integrations,
and ISV partnerships. We chose the name Koi as it means "Any" (for any data format, unstructured or structured) in Hindi as well as accurately represents the underlying technology i.e. 
K(C)ognitive Optical Imaging.Our founders are alums of XPO Logistics, EY, PwC, Mu Sigma, Oracle, Blue Yonder/JDA/i2, and Capgemini. We offer deep expertise in Artificial Intelligence, EDGE Computing, Image Processing, Algorithms, Data Science, Logistics, Trade, Transportation, Maritime, and Supply Chain.

Our platform capabilities are backed by multiple patents.KoiReader Technologies is a GLOBAL AWARD WINNING FreightTech 100 company with deep industry and operations footprint. Our core product is AIoT-powered Hyperautomation Platform that includes Document Digitization and Machine Vision."""
               
               
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

