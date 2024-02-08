import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

# Load Data Function
def load_data(csv_file_path, sep=';', index_col=None):
    """Loads data from a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_file_path, sep=sep, index_col=index_col)
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return None

# Collaborative Filtering Function
def collaborative_filtering(ratings_df):
    """Performs collaborative filtering using the SVD algorithm."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'product_id', 'rating']], reader)
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.04]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    algo = gs.best_estimator['rmse']
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    return algo

# Precision and Recall Calculation
def precision_recall_at_k(predictions, k=5, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precision = dict()
    recall = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
        precision[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recall[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precision, recall

# Content-Based Similarity Calculation
def calculate_similarity(products):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Hybrid Recommendation
def hybrid_recommendation(user_id, products, ratings, algo, k=5):
    cf_predictions = [algo.predict(user_id, pid).est for pid in products['product_id']]
    products['cf_score'] = cf_predictions
    cbf_similarity = calculate_similarity(products)
    products['cbf_score'] = cbf_similarity.mean(axis=1)
    products['hybrid_score'] = (products['cf_score'] + products['cbf_score']) / 2
    recommended_products = products.sort_values('hybrid_score', ascending=False).head(k)
    products.drop(columns=['cf_score', 'cbf_score', 'hybrid_score'], inplace=True, errors='ignore')
    return recommended_products

# Streamlit App
st.title("Makeup Product Recommendation System")

# Load and preprocess data
product_metadata_path = 'Makeup_Products_Metadata.csv'
user_ratings_path = 'User_review_data.csv'
products_df = load_data(product_metadata_path, sep=';')
ratings_df = load_data(user_ratings_path, sep=';', index_col='User')

# Preprocess ratings data
ratings = ratings_df.reset_index().melt(id_vars='User', var_name='Item', value_name='Rating')
ratings = ratings[ratings['Rating'] > 0]
ratings['user_id'] = ratings['User'].astype('category').cat.codes
ratings.rename(columns={'User': 'user_name', 'Item': 'product_id', 'Rating': 'rating'}, inplace=True)
user_name_to_id = pd.Series(ratings['user_id'].values, index=ratings['user_name'].str.lower()).to_dict()

# Model training
algo = collaborative_filtering(ratings)

# User interaction for recommendations
user_input = st.text_input("Type 'guest' to continue as guest or enter your username for personalized recommendations:")

if user_input:
    if user_input.lower() == 'guest':
        st.write("Guest recommendations (showing top products)...")
        # Implement logic for guest recommendations or popular items here
    else:
        st.write(f"Welcome back, {user_input}! Personalized recommendations:")
        user_id = user_name_to_id.get(user_input.lower())
        if user_id is not None:
            recommended_products = hybrid_recommendation(user_id, products_df.copy(), ratings, algo, 5)
            st.dataframe(recommended_products[['name', 'description']])
        else:
            st.write("User not found. Showing top rated products instead.")
            # Implement logic for showing top products
