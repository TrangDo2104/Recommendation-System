import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

# Assuming all your functions are defined here as in your script
def load_data(csv_file_path, sep=';', index_col=None):
    """Loads data from a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_file_path, sep=sep, index_col=index_col)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading the data: {e}")
        return None

# Load product metadata and user ratings data
product_metadata_path = 'Makeup_Products_Metadata.csv'  # Update path as necessary
user_ratings_path = 'User_review_data.csv'  # Update path as necessary

products_df = load_data(product_metadata_path, sep=';')
ratings_df = load_data(user_ratings_path, sep=';', index_col='User')

missing_values_count = products_df.isnull().sum()
print(missing_values_count)

products= products_df[['Product ID','Product Name', 'Product Price [SEK]','Product Description']]
products.columns = ['product_id', 'name', 'price', 'description']
products.head()

ratings = ratings_df.reset_index().melt(id_vars='User', var_name='Item', value_name='Rating')
ratings = ratings[ratings['Rating'] > 0]

# Convert 'user_name' to a categorical type and then to numerical codes
ratings['user_id'] = ratings['User'].astype('category').cat.codes
ratings.columns = ['user_name', 'product_id', 'rating', 'user_id']
user_name_to_id = pd.Series(ratings['user_id'].values, index=ratings['user_name'].str.lower()).to_dict()
ratings.head()

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

# Ensure pandas apply map is available for this example
pd.set_option('mode.chained_assignment', None)

# Collaborative Filtering with GridSearchCV
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'product_id', 'rating']], reader)

# Split data into training and test set
trainset, testset = train_test_split(data, test_size=0.25)

# GridSearchCV for SVD hyperparameters
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.04]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Best SVD model
algo = gs.best_estimator['rmse']

# Re-train on the full dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Evaluate Precision@K and Recall@K
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

# Testing CF model on test set and calculating metrics
predictions = algo.test(testset)
precision, recall = precision_recall_at_k(predictions)

avg_precision = np.mean(list(precision.values()))
avg_recall = np.mean(list(recall.values()))

print(f'Average Precision: {avg_precision:.2f}')
print(f'Average Recall: {avg_recall:.2f}')

# Content-Based Filtering
def calculate_similarity(products):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Hybrid Recommendation Function
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

# Find similar products by description
def find_similar_products_by_description(query, products, k=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-k:][::-1]
    return products.iloc[top_indices][['product_id', 'name']]

# Initialize session state variables
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''
if 'satisfied' not in st.session_state:
    st.session_state['satisfied'] = None
if 'follow_up_query' not in st.session_state:
    st.session_state['follow_up_query'] = ''
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'show_similar_products' not in st.session_state:
    st.session_state.show_similar_products = False

# Load data
products_df = load_data(product_metadata_path, sep=';')
ratings_df = load_data(user_ratings_path, sep=';', index_col='User')

# UI for user input
st.title("Makeup Product Recommendation System")
user_input = st.text_input("Type 'guest' to continue as guest or enter your username for personalized recommendations:", key='user_input')

# Function to display recommendations
def display_recommendations(user_input):
    user_id = user_name_to_id.get(user_input.lower())
    if user_id is not None:
        recommended_products = hybrid_recommendation(user_id, products_df.copy(), ratings_df, algo, 5)
        st.write(f"Welcome back, {user_input.capitalize()}! Here are your personalized recommendations:")
        st.dataframe(recommended_products[['name', 'description']])
    else:
        st.write("Guest or new user recommendations (placeholder)")
    st.session_state.show_recommendations = True

# Check for user input to display recommendations
if user_input and not st.session_state.satisfied:
    display_recommendations(user_input)

# Satisfaction check
if st.session_state.show_recommendations:
    satisfied = st.radio("Are you happy with these recommendations?", ("Yes", "No"), key='satisfied')

# Follow-up action based on satisfaction
if satisfied == "No":
    follow_up_query = st.text_input("Find your product here:", key='follow_up_query')
    if follow_up_query:
        similar_products = find_similar_products_by_description(follow_up_query, products_df, 5)
        st.write("Top 5 relevant products to your description input:")
        st.dataframe(similar_products[['product_id', 'name']])
        st.session_state.show_similar_products = True

# Resetting the interaction
if st.button("Start Over"):
    st.session_state.user_input = ''
    st.session_state.satisfied = None
    st.session_state.follow_up_query = ''
    st.session_state.show_recommendations = False
    st.session_state.show_similar_products = False
    st.experimental_rerun()
