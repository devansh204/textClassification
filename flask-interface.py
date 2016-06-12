from flask import Flask
from flask import request
from flask import render_template
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import PorterStemmer
import pickle
from collections import Counter
from pybrain.datasets import ClassificationDataSet

app = Flask(__name__)
app.secret_key = 'some_secret'

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
			normalised_data.append(values[0])
	return( " ".join( normalised_data ))

@app.route('/')
def my_form():
	app.logger.info('Info')
	return render_template('myform.html')

@app.route('/', methods=['POST'])
def my_form_post():

    text = request.form['text']
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', '')

    fileObject1 = open('ANNDUMP','r')
    fnn = pickle.load(fileObject1)
    fileObject2 = open('VOCABDUMP','r')
    vocab = pickle.load(fileObject2)
    fileObject3 = open('STDSCALER','r')
    scaler = pickle.load(fileObject3)

    review_nor = normalise_review(text)
    review_nor = nltk.word_tokenize(review_nor)

    string = ""
    string += str(review_nor)
    token_count = Counter(review_nor)

    feature_list = []
    for words in vocab:
    	feature_list.append(token_count[words])
    feature_list =  scaler.transform(feature_list)

    griddata = ClassificationDataSet(1600,1, nb_classes=2)
    griddata.addSample(tuple(feature_list), [0])
    griddata._convertToOneOfMany()

    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)
    string += "\nThis review belongs to the class "
    if (out[0] == 0):
    	string += "NEGATIVE"
    if (out[0] == 1):
    	string += "POSITIVE"

    return string

if __name__ == '__main__':
    app.run(port=5023)