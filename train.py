from collections import Counter
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.validation import ModuleValidator
from pybrain.tools.validation import CrossValidator
from sklearn import preprocessing

import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import pickle

# This function takes the raw text as the param, and returns the 
# normalised text back
def normalise_review( raw_review ):

	stemmer=PorterStemmer()
	raw_review = re.sub("[^a-zA-Z]", " ", raw_review)
	raw_review = raw_review.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in raw_review if not w in stops]
	meaningful_words = nltk.pos_tag(meaningful_words)
	# We consider all the adjectives and adverbs
	tag_list = ['JJ','JJR','JJS','RB','RBS','RBR']
	normalised_data = []
	for values in meaningful_words:
		if values[1] in tag_list:
			x = stemmer.stem(values[0])
			x = str(x)
			normalised_data.append(x)
	return( " ".join( normalised_data ))

clean_train_reviews = []
rating_list = []

#Reading the CSV file for reviews
fields = ['rating','review']
reviews = pd.read_csv('reviews.csv',skipinitialspace = True, usecols=fields)
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

# Here we initialize CountVectorizer object, scikit-learn's
# bag of words tool 
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, \
                             preprocessor = None, stop_words = None, \
                             max_features = 1600)

# fit_transform transforms our data into feature vectors.
# Here we extract the top features (max_features)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
vocab = vectorizer.get_feature_names()

# We represent every review in the terms of (max_features)
# only, ie, the top features which exist in vocab
list_features = []
for review in clean_train_reviews:
	review_token = nltk.word_tokenize(review)
	review_count = Counter(review_token)
	temp_list = []

	for words in vocab:
		temp_list.append(review_count[words])

	list_features.append(temp_list)

# We standardise them in the range -1 to 1
scaler = preprocessing.StandardScaler().fit(list_features)
list_features = np.array(list_features)

# The standardized object is stored
fileObject = open('STDSCALER', 'wb')
pickle.dump(scaler, fileObject)
fileObject.close()

list_features =  scaler.transform(list_features)

# We now start building our network. 
alldata = ClassificationDataSet(1600, 1, nb_classes=2)
for i in range (0,len(rating_list)):
	alldata.addSample(list_features[i], rating_list[i])

# Our dataset is divided, with 80% as training data and 
# 20% as test data
tstdata_temp, trndata_temp = alldata.splitWithProportion(0.2)

tstdata = ClassificationDataSet(1600, 1, nb_classes=2)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(1600, 1, nb_classes=2)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

fnn = buildNetwork( trndata.indim,120,trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.00001)
modval = ModuleValidator()

# We define the number of iterations we want to train our model.
for i in range(100):
	trainer.trainEpochs(1)
	trnresult = percentError(trainer.testOnClassData(dataset=trndata),trndata['class'])
	print "epoch : " , trainer.totalepochs," train error: " , trnresult

# We validate our model by applying the n-folds technique and check the Mean Square Error
cv = CrossValidator( trainer, trndata, n_folds=5, valfunc=modval.MSE )
print "MSE %f at loop %i"%(cv.validate(),i)

# Finally we test our data on the model we built
perror = percentError(trainer.testOnClassData(dataset=tstdata),tstdata['class'])
print " Percent error on test data is - ",100.0 - perror

# We also take a dump of the model and the top features
fileObject2 = open('ANNDUMP', 'wb')
pickle.dump(fnn, fileObject2)
fileObject2.close()

fObject = open('VOCABDUMP','wb')
pickle.dump(vocab, fObject)
fObject.close()