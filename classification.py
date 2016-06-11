import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from collections import Counter
import pickle

def normalise_review( raw_review ):

	stemmer=PorterStemmer()
	raw_review = re.sub("[^a-zA-Z]", " ", raw_review)
	raw_review = raw_review.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in raw_review if not w in stops]
	meaningful_words = nltk.pos_tag(meaningful_words)
	tag_list = ['JJ','JJR','JJS','RB','RBS','RBR']
	normalised_data = []
	for values in meaningful_words:
		if values[1] in tag_list:
			x = stemmer.stem(values[0])
			x = str(x)
			normalised_data.append(x)
	return( " ".join( normalised_data ))

fields = ['rating','review']
reviews = pd.read_csv('reviews.csv',skipinitialspace = True, usecols=fields)

clean_train_reviews = []
rating_list = []

print "Collecting Reviews."
for i in range (0, len(reviews.rating)):
	try:
		rating = float(reviews.rating[i])
		review = str(reviews.review[i])

		normalised_data = normalise_review(review)

		if (len(normalised_data) != 0):
			clean_train_reviews.append(normalised_data)
			rating_list.append(rating)
	except:
		pass

print clean_train_reviews
print rating_list

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, \
                             preprocessor = None, stop_words = None, \
                             max_features = 1600)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()

list_features = []

for review in clean_train_reviews:
	review_token = nltk.word_tokenize(review)
	review_count = Counter(review_token)
	temp_list = []
	for words in vocab:
		temp_list.append(review_count[words])

	list_features.append(temp_list)

scaler = preprocessing.StandardScaler().fit(list_features)
list_features = np.array(list_features)

fileObject = open('STDSCALER', 'wb')
pickle.dump(scaler, fileObject2)
fileObject.close()