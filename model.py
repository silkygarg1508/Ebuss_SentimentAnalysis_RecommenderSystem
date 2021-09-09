from flask import Flask, jsonify,  request, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# load the model from disk
filename = 'models/logistic_model.pkl'
model = pickle.load(open(filename, 'rb'))

#loading tfidf vectorizer pkl file
filename = 'models/tfidfvectorizer.pkl'
tfidf = pickle.load(open(filename, 'rb'))

#reading reviews file
reviews = pd.read_csv('data/reviews.csv')

#reading user final rating whihc has the rating for all the users in the dataset after having found the user user correlation amongst all the users
#selected user-user correlation as it gave less rmse as compared to item-item based
user_final_rating = pd.read_csv('data/user_final_rating.csv')
user_final_rating.set_index("reviews_username_encoded",inplace=True)
user_final_rating.T.index.name = 'name_encoded'

#Function to recommend top 20 products for the given user
def recommendation(user):
	top20_user = user_final_rating.loc[user].sort_values(ascending=False)[0:20]
	top20_user = top20_user.to_frame()
	top20_user.reset_index(inplace=True)
	top20_user.columns = ['name_encoded','weigthed_rating']
	top20_user['name_encoded'] = top20_user['name_encoded'].apply(lambda x: int(x))
	#Merging top 5 items with the reviews df so as to get their details 
	top20_user = pd.merge(top20_user,reviews,left_on='name_encoded',right_on='name_encoded', how = 'left')
	#Dropping irrelevant columns to avoid duplicacy
	top20_user.drop(['weigthed_rating','reviews_date','reviews_doRecommend','reviews_rating','reviews_text','reviews_title','reviews_username','user_sentiment','reviews_combined','reviews_username_encoded'],axis =1,inplace=True)
	top20_user.drop_duplicates(inplace=True)
	
	return top20_user
	
#Function to recommend top 5 products based on sentiment analysis model for the given user from the top 20 recommended products
def sentiment(user_input):
	#Finding out user id corresponding to the username for further processing
	user_input = user_input.lower()
	#Creating a list of top5 products to be sent to output
	list_of_top5_products=[]
	if user_input in list(reviews['reviews_username']):
		user = (reviews['reviews_username_encoded'].loc[reviews['reviews_username'] == user_input]).drop_duplicates().reset_index(drop=True)[0]
		top20_user = recommendation(user)
		top5_user = pd.merge(top20_user,reviews,left_on='name_encoded',right_on='name_encoded',how = 'left')
		top5_user = top5_user[['name_x','reviews_combined']]
		top5_user.rename(columns={'name_x':'name'},inplace=True)
		#using tfidf vectorizer to transform
		X = tfidf.transform(top5_user.reviews_combined).toarray()
		top5_user['predicted_sentiment'] = model.predict(X)
		#dropping reviews_combined as we don't need it now
		del top5_user['reviews_combined']
		#top5_user['positive_sentiment_percentage'] = top5_user.groupby(['name']).mean()
		top5_user = top5_user.groupby(['name']).mean()
		top5_user.reset_index(inplace=True)
		#Sorting the obtained dataframe based on predicted_sentiment and displaying the top5 results
		top5_user = top5_user.sort_values(by=['predicted_sentiment'],ascending=False)
		top5_user = top5_user[0:5]
		#Displaying top 5 items 
		top5_user.reset_index(inplace=True)
		for item in top5_user['name']:
			list_of_top5_products.append(item)
	return list_of_top5_products
