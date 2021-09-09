# Ebuss_SentimentAnalysis_RecommenderSystem
# Added reviews.csv files which contains the cleaned data from sample30.csv and also the encoded columns which are to be used by recommendation engine
# Added user_final_rating.csv which contains the ratings for all the users in the dataset after having found the user user correlation amongst all the users
# selected user-user correlation as it gave less rmse(2.5) as compared to item-item based(3.5)
# Since Logistic Regression has given best Recall score for both positive(0.88) and negative sentiment(0.80), hence we will use logistic model for doing sentiment analysis on top 20 recommendations
# Using tfidf Vectorizer and hence it's pickle file for transformation

# Some test users present in the dataset
# joshua,rebecca,walker557

# Some randome users not present in the dataset
# silky
