from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import re
import string
import pandas as pd
import numpy as np


def clean_text_round1 (input_message):
    ''' remove: urls;
    lowercase text;
    remove: punctuations and numbers'''

    remove_urls = re.sub('https?://[A-Za-z0-9./]+','',input_message)
    remove_alphanumeric= re.sub('\w*\d\w*', ' ', remove_urls)
    punc_lower = re.sub('[%s]' % re.escape(string.punctuation), '', remove_alphanumeric.lower())
    # consider replacing ' ' with '' to remove space between I ' m to I'm
    return (punc_lower)

def clean_text_round2 (input_message):
    input_message = input_message.replace('flighted', 'flight').replace('flightled', 'flight').replace('flights', 'flight').replace('\n', ' ')
    return input_message



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/sentiment',methods=['POST'])
def predict():

	logreg_model = open('/Users/evaxu/Desktop/Metis/Local_Projects_files/project_4/LogReg_model_weighted.pkl','rb')
	logreg = joblib.load(logreg_model)
	tfidf_binomial_fit = open('/Users/evaxu/Desktop/Metis/Local_Projects_files/project_4/tfidf_fit.pkl','rb')
	tfidf_binomial_fit = joblib.load(tfidf_binomial_fit)

	if request.method == 'POST':
		message = request.form['message']
		message1 = clean_text_round1(message)
		message2 = clean_text_round2(message1)
		data = [message2]
		vect = tfidf_binomial_fit.transform(data).toarray()
		my_prediction = int(logreg.predict(vect))

	tfidf_model_ngrams = open('/Users/evaxu/Desktop/Metis/Local_Projects_files/project_4/tfidf_model_ngrams.pkl','rb')
	tfidf_model_ngrams = joblib.load(tfidf_model_ngrams)
	nmf_model_ngrams = open('/Users/evaxu/Desktop/Metis/Local_Projects_files/project_4/nmf_model_ngrams.pkl','rb')
	nmf_model_ngrams = joblib.load(nmf_model_ngrams)

	if my_prediction == 0:
	    array = tfidf_model_ngrams[0].transform(data).toarray()
	    topicprob = nmf_model_ngrams[0].transform(array)
	    maxElement = np.amax(topicprob[0])
	    index = np.where(topicprob[0]==maxElement)
	    index = index[0].tolist()
	    index = index[0]+1
	    negative_topics = { 1: 'Flight Cancellations' , 2: 'Customer Service Issues', 3: 'Flight Delays', 4: 'Flight Cancellations' , 5: 'Unknown', 6: 'Flight Delays', 7: 'Customer Service Issues'}
	    topic = negative_topics[index]

	if my_prediction == 1:
	    array = tfidf_model_ngrams[1].transform(data).toarray()
	    topicprob = nmf_model_ngrams[1].transform(array)
	    maxElement = np.amax(topicprob[0])
	    index = np.where(topicprob[0]==maxElement)
	    index = index[0].tolist()
	    index = index[0]+1
	    positive_topics = { 1: 'Unknown' , 2: 'Unknown', 3: 'Inquiries', 4: 'Customer Satisfaction', 5: 'Unknown' , 6: 'Inquiries', 7: 'Customer Satisfaction'}
	    topic = positive_topics[index]

	return render_template('home.html',prediction = my_prediction, topic = topic, input = message)







if __name__ == '__main__':
	app.run(debug=True)
