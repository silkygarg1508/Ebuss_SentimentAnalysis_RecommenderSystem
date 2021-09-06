from flask import Flask, jsonify,  request, render_template
#from sklearn.externals import joblib
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
# load the model from disk
filename = 'models/logistic_model.pkl'
model = pickle.load(open(filename, 'rb'))
#model_load = joblib.load("./models/logistic_model.pkl")

#reading reviews file
reviews = pd.read_csv('data/reviews.csv')

#reading user final rating
user_final_rating = pd.read_csv('data/user_final_rating.csv')
user_final_rating.set_index("reviews_username_encoded",inplace=True)
user_final_rating.T.index.name = 'name_encoded'

#reading item final rating
item_final_rating = pd.read_csv('data/item_final_rating.csv')
item_final_rating.set_index("reviews_username_encoded",inplace=True)
item_final_rating.T.index.name = 'name_encoded'

def recommendation(user):
	#user_input = int(user_input)
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
	
def sentiment(user_input):
	#Finding out user id corresponding to the username for further processing
	user = reviews['reviews_username_encoded'].loc[reviews['reviews_username'] == user_input][0]
	top20_user = recommendation(user)
	top5_user = pd.merge(top20_user,reviews,left_on='name_encoded',right_on='name_encoded',how = 'left')
	top5_user = top5_user[['name_x','reviews_combined']]
	top5_user.rename(columns={'name_x':'name'},inplace=True)
	#initializing tfidf vectorizer
	tfidf = TfidfVectorizer(max_features = 2500)
	X = tfidf.fit_transform(top5_user.reviews_combined).toarray()
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
	list_of_top5_products=[]
	for item in top5_user['name']:
		list_of_top5_products.append(item)
	return list_of_top5_products
	
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/recommend", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_input = request.form['Username']
        output = sentiment(user_input)
        return render_template('index.html', prediction_text=output)
    else :
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
