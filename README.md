# Sentiment Analysis of IMDB Reviews Using Neural Networks

There is a set of 25,000 reviews from IMDB with a given sentiment. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1.

The set of 25,000 reviews is split into training (80% of total) and testing (20% of total) data. We use the python library Pybrain to implement the Neural Network model.

* train.py - This file contains the code to train and store our model
* reviews.csv - This file contains the reviews (raw text) along with their sentiment
* flask-interface.py - This is the minimalistic interface using which we can test any review manually for it's predicted sentiment
