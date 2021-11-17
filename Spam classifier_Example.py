# Importing the Required Libraries ########

import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import WordNetLemmatizer

## Importing the Required Data #######

messages = pd.read_csv('SMSSpamCollection.txt',sep = '\t',names = ['Label','Message'])

## nltk Technique ###########
wordnet = WordNetLemmatizer()

## Cleaning the Text ####

corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['Message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

## Creating the TFIDF Model #######

from sklearn.feature_extraction.text import TfidfVectorizer
TFID = TfidfVectorizer(max_features = 3000)
x = TFID.fit_transform(corpus).toarray()

# Converting the Dependent Variable into Numerical Format #####

y = pd.get_dummies(messages['Label'])
y = y.iloc[:,1].values

## Training the Model using the Navis bayes Classifier

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB()
spam_detect.fit(x_train,y_train)

y_pred = spam_detect.predict(x_test)

## Metrics ###

from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test,y_pred)

acc_score = accuracy_score(y_test,y_pred)