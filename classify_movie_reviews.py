import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

#add your directory path where training dataset is available
train = pd.read_csv('F:\Semester 5\AI\Labs\lab5\labeledTrainData.tsv\labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

vectorizer = CountVectorizer(analyzer = "word", \
tokenizer = None, \
preprocessor = None, \
stop_words = None, \
max_features = 5000)

#cleaning the dataset
def review_to_words(raw_review):
    #removing HTML tags
    temp1 = BeautifulSoup(raw_review)
    temp2 = temp1.get_text()
    #removing non-letters
    temp3 = re.sub(r'[^a-zA-Z]'," ",temp2)
    #lowercase & tokenization
    temp4 = temp3.lower().split()
    #Remove stops words
    stops = set(stopwords.words("english"))
    words = [w for w in temp4 if not w in stops]
    #Joint back and return the joined sentence
    sentence = " ".join(words);
    return sentence

#initializing data structures
train_data = []
for x in range (0, 20000):
    train_data.append(review_to_words(train["review"][x]))
mylabels = []
for x in range (0, 20000):
    mylabels.append(train["sentiment"][x])
test_data = []
old_senti_of_test = []
for x in range (20001, 25000):
    test_data.append(review_to_words(train["review"][x]))
    old_senti_of_test.append(train["sentiment"][x])


X = vectorizer.fit_transform(train_data)
X = X.toarray()
clf = MultinomialNB(alpha=0.00001)
clf.fit(X, np.array(mylabels))

tX = vectorizer.transform(test_data).toarray()

new_senti = clf.predict(tX)
count = 0
for x in range (0, 4999):
    if (old_senti_of_test[x] == new_senti[x]):
        count += 1

#print (clf.predict_proba(tX))
acc = (count / 5000) * 100
print("Accuracy: ", acc)