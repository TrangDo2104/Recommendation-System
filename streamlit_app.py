import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from collections import defaultdict

# Streamlit UI components
st.title('Hybrid Recommendation System')

# Define your data loading and processing functions
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

products_dfs = load_data(product_metadata_path, sep=';')
ratings_dfs = load_data(user_ratings_path, sep=';', index_col='User')

products_df= products_dfs[['Product ID','Product Name', 'Product Price [SEK]','Product Description']]
products_df.columns = ['product_id', 'name', 'price', 'description']
products_df.head()

ratings_df = ratings_dfs.reset_index().melt(id_vars='User', var_name='Item', value_name='Rating')
ratings_df = ratings_df[ratings_df['Rating'] > 0]

# Convert 'user_name' to a categorical type and then to numerical codes
ratings_df['user_id'] = ratings_df['User'].astype('category').cat.codes
ratings_df.columns = ['user_name', 'product_id', 'rating', 'user_id']
user_name_to_id = pd.Series(ratings_df['user_id'].values, index=ratings_df['user_name'].str.lower()).to_dict()

def collaborative_filtering(ratings_df):
    """Adapts the collaborative filtering process for Streamlit."""
    st.write("Starting Collaborative Filtering with SVD algorithm...")

    # Data preparation
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'product_id', 'rating']], reader)
    
    # Split data into training and test set
    trainset, testset = train_test_split(data, test_size=0.25)

    # GridSearchCV for SVD hyperparameters
    st.write("Tuning hyperparameters...")
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.04]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)

    # Best SVD model
    algo = gs.best_estimator['rmse']
    st.write(f"Best hyperparameters: {gs.best_params['rmse']}")

    # Re-train on the full dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # The function should return 'algo' which is the trained model object.
    return algo

def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """Calculates precision and recall at k for given predictions."""
    # Identical to the original function, no changes required here.
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


# Calculate similarity for content-based filtering
def calculate_similarity(products_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def hybrid_recommendation(user_id, products, ratings, algo, k=5, product_name=None):
    # Work with a copy to avoid modifying the original DataFrame
    products_copy = products.copy()
    
    # Check if user_id exists to provide CF predictions
    if user_id is not None:
        cf_predictions = [algo.predict(user_id, pid).est for pid in products_copy['product_id']]
        products_copy['cf_score'] = cf_predictions
    else:
        products_copy['cf_score'] = 0  # Default to 0 if no user_id
    
    # Calculate CBF similarity if product_name is provided
    if product_name:
        cosine_sim = calculate_similarity(products_copy)
        idx = products_copy.index[products_copy['name'].str.lower() == product_name.lower()].tolist()
        if idx:
            sim_scores = list(enumerate(cosine_sim[idx[0]]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]  # Ignore the first one as it is the product itself
            product_indices = [i[0] for i in sim_scores]
            products_copy['cbf_score'] = 0
            products_copy.loc[product_indices, 'cbf_score'] = [score[1] for score in sim_scores]
        else:
            products_copy['cbf_score'] = 0  # Default to 0 if no product_name match
    else:
        # Calculate CBF similarity based on existing product descriptions
        cbf_similarity = calculate_similarity(products_copy)
        products_copy['cbf_score'] = cbf_similarity.mean(axis=1)
    
    # Hybrid score: average of CF and CBF scores
    products_copy['hybrid_score'] = (products_copy['cf_score'] + products_copy['cbf_score']) / 2
    
    # Sort and clean up
    recommended_products = products_copy.sort_values('hybrid_score', ascending=False).head(k)
    
    return recommended_products

    
# Train the model
algo = collaborative_filtering(ratings_df)

def find_similar_products_by_description(query, products, k=5):
    """Find similar products based on description."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-k:][::-1]
    return products.iloc[top_indices][['product_id', 'name']]

def personalized_recommendation(user_input, products, ratings, algo, user_name_to_id, k=5):
    """Generate personalized product recommendations."""
    user_id = user_name_to_id.get(user_input.lower())
    if user_id is not None:
        st.write(f"Welcome back, {user_input.capitalize()}! Here are your personalized recommendations:")
        recommended_products = hybrid_recommendation(user_id, products.copy(), ratings, algo, k)
        st.write(recommended_products[['name', 'hybrid_score']])
    else:
        st.write("User not found or you're a new user. Let's find some products for you.")

def main_interaction_streamlit(products, ratings, algo, user_name_to_id):
    """Main interaction flow adapted for Streamlit."""
    user_input = st.text_input("Enter your name for personalized recommendations or enter a product description below:", '')
    
    if user_input:
        personalized_recommendation(user_input, products, ratings, algo, user_name_to_id, 5)
    
    query = st.text_input("What are you looking for? Enter a product description or name:", '')
    
    if query:
        similar_products = find_similar_products_by_description(query, products, 5)
        if not similar_products.empty:
            st.write("Top relevant products to your description input:")
            st.table(similar_products)
        else:
            st.write("No similar products found.")

# Place this within your Streamlit app code structure
# Ensure all necessary functions and data are defined and available
if 'restart' not in st.session_state:
    st.session_state['restart'] = False

if not st.session_state['restart']:
    main_interaction_streamlit(products_df, ratings_df, algo, user_name_to_id)
