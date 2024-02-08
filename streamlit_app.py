import pandas as pd
import numpy as np
import streamlit as st
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

@st.cache
def load_data(csv_file_path, sep=';', index_col=None):
    """Loads data from a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_file_path, sep=sep, index_col=index_col)
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")

@st.cache
def collaborative_filtering(ratings_df):
    """Performs collaborative filtering using the SVD algorithm."""
    # Data preparation
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.04]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    
    # Best model and retraining
    algo = gs.best_estimator['rmse']
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    return algo

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

def calculate_similarity(products):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def hybrid_recommendation(user_id, products, ratings, algo, k=5):
    # Logic as provided
    # Check if user_id exists to provide CF predictions
    cf_predictions = [algo.predict(user_id, pid).est for pid in products['product_id']]
    products['cf_score'] = cf_predictions
    
    # Calculate CBF similarity
    cbf_similarity = calculate_similarity(products)
    products['cbf_score'] = cbf_similarity.mean(axis=1)
    
    # Hybrid score
    products['hybrid_score'] = (products['cf_score'] + products['cbf_score']) / 2
    
    # Sort and clean up
    recommended_products = products.sort_values('hybrid_score', ascending=False).head(k)
    products.drop(columns=['cf_score', 'cbf_score', 'hybrid_score'], inplace=True, errors='ignore')
    
    return recommended_products

def find_similar_products_by_description(query, products, k=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-k:][::-1]
    return products.iloc[top_indices][['product_id', 'name']]

def personalized_recommendation(user_input, products, ratings, algo, user_name_to_id, k=5):
    user_id = user_name_to_id.get(user_input.lower())
    if user_id is not None:
        st.write(f"Welcome back, {user_input.capitalize()}! Here are your personalized recommendations:")
        recommended_products = hybrid_recommendation(user_id, products.copy(), ratings, algo, k)
        st.write(recommended_products)
        
        satisfied = st.radio("Are you happy with these recommendations?", ('Yes', 'No'))
        if satisfied == 'No':
            query = st.text_input("Find your product here:")
            st.write("Top 5 relevant products to your description input:")
            similar_products = find_similar_products_by_description(query, products, k)
            st.write(similar_products)
    else:
        st.write("User not found or you're a new user. Let's find some products for you.")
        query = st.text_input("What are you looking for?")
        st.write("Top 5 relevant products to your description input:")
        similar_products = find_similar_products_by_description(query, products, k)
        st.write(similar_products)

def main():
    st.title('Product Recommendation System')

    # Load data
    product_metadata_path = 'Makeup_Products_Metadata.csv'
    user_ratings_path = 'User_review_data.csv'
    products_df = load_data(product_metadata_path, sep=';')
    ratings_df = load_data(user_ratings_path, sep=';', index_col='User')

    # Data preprocessing
    products = products_df[['Product ID','Product Name', 'Product Price [SEK]','Product Description']]
    products.columns = ['product_id', 'name', 'price', 'description']

    ratings = ratings_df.reset_index().melt(id_vars='User', var_name='Item', value_name='Rating')
    ratings = ratings[ratings['Rating'] > 0]
    ratings['user_id'] = ratings['User'].astype('category').cat.codes
    ratings.columns = ['user_name', 'product_id', 'rating', 'user_id']
    user_name_to_id = pd.Series(ratings['user_id'].values, index=ratings['user_name'].str.lower()).to_dict()

    # Collaborative filtering
    algo = collaborative_filtering(ratings)

    # Main interaction flow
    user_input = st.text_input("Type 'guest' to continue as guest or enter your name for personalized recommendations")
    if st.button('Get Recommendations'):
        personalized_recommendation(user_input, products, ratings, algo, user_name_to_id, 5)

if __name__ == "__main__":
    main()
