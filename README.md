# Ebuss_SentimentAnalysis_RecommenderSystem
#### 1) Added reviews.csv files which contains the cleaned data from sample30.csv and also the encoded columns which are to be used by recommendation engine
#### 2) Added user_final_rating.csv which contains the ratings for all the users in the dataset after having found the user user correlation amongst all the users
#### 3) selected user-user correlation as it gave less rmse(2.5) as compared to item-item based(3.5)
#### 4) Since Logistic Regression has given best Recall score for both positive(0.88) and negative sentiment(0.80), hence we will use logistic model for doing sentiment analysis on top 20 recommendations
#### 5) Using tfidf Vectorizer and hence it's pickle file for transformation

#### 6) Some test users present in the dataset
joshua,rebecca,walker557

#### 7) Some random users not present in the dataset
silky
