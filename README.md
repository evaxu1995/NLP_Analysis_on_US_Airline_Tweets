# NLP-Analysis-on-US-Airline-Tweets

I used a natural language processing (NLP) approach to understand the topics underlying customer tweets within a Twitter US Airlines dataset. I used a semi-supervised classification method where I first classified the sentiments of the tweets as positive/neutral or negative using a Logistics Regression model. Then, I overlayed topic modeling/dimensionality reduction techniques to extract unique topics for each sentiment. Finally, I created an application using Twitter’s API and Flask to deploy my model to production.

## Project Intro/Objective

Today, social media data in the form of Twitter, Facebook and Instagram posts have become a place to instantaneously know of the general perception. People have openly started sharing their opinions and customers have changed their way of engaging with brands online.  One of the largest industries that has been affected by this shift to online customer engagement is the US Airline industry. <br>
When it comes to customer service, travelers are increasingly skipping calls to the airlines and are instead taking their requests to Twitter and Facebook and airlines are responding by expanding their social media staff to aid travelers. With rising online customer inquiries, it makes a lot of sense to have a system in place that is capable of ingesting this incessant stream of data, organizing them into categories, and then directing the customers to staff responsible for those categories so that they can answer those questions effectively and quickly. To do this, I wanted to create a classifier that would be able to identify whether customer tweets are negative or positive/neutral or, and extracts the main topics associated with their sentiment. Negative tweets can provide insight to issues that are bothering the customer, and the positive/neutral tweets can shed light on the effectiveness of the staff in resolving the issues. Through this analysis, US Airlines can better service their customers as well as evaluate their own performance. 

## Datasets Used
•	February 2015 Major US Airlines Tweets from Kaggle <br>

## Methods Used
•	Exploratory Data Analysis (EDA) <br>
•	Data Visualization with Tableau <br>
•	Natural Language Processing (NLP) <br>
•	Topic Modeling <br>
•	Sentiment Analysis <br>
•	Feature Engineering <br>
•	Application with Flask <br>

## Notable Technologies Used
•	Python 3, Jupyter Notebook <br>
•	Nltk, Spacy, Scikit-learn # NLP Text Processing <br>
•	CountVectorizer, TfidfVectorizer, NMF # Topic Modeling <br>
•	Logistic Regression # Sentiment Analysis Model <br>
•	Pandas, Numpy, Matplotlib, Seaborn, Tableau, Flask # Data Processing/Visualization tools <br>
•	etc. <br>

## Main Analysis Threads
•	Tweet Cleaning - Tweet cleaning by removing hastags, retweets, @s, links, special characters, selecting 500,000 customer tweets to analyze <br>
•	Tweet Tokenization, and Vectorization - Tokenization through lowercasing and removal of numbers, punctuation and stopwords; term frequency-inverse document frequency count vectorization using ScikitLearn's TfidfVectorizer submodule <br>
•	Sentiment Analysis - Used Logistic Regression model to classify the overall sentiment for each tweet based on features extracted from TF-IDF <br>
•	Topic Modeling / Dimensionality Reduction - Ran topic modeling for each sentiment using non-negative matrix factorization (NMF); extracted 4 unique topics for negative tweets, and 3 unique topics for positive/neutral tweets <br>
•	Application – Created a Flask application that outputs the sentiment and topic behind any customer tweet <br>
