import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from collections import defaultdict

# Custom CSS for styling
st.markdown("""
    <style>
        .full-width {
            width: 100%;
        }
        .success-message {
            color: green;
            font-weight: bold;
        }
        .warning-message {
            color: orange;
            font-weight: bold;
        }
        .error-message {
            color: red;
            font-weight: bold;
        }
        .content-container {
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
        }
        .recommendation-container {
            padding: 20px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            color: #333333;
        }
        .sub-header {
            font-size: 18px;
            font-weight: bold;
            color: #666666;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI components
st.title('🌈 Hybrid Recommendation System')

# Define your data loading and processing functions
def load_data(csv_file_path, sep=';', index_col=None):
    """Loads data from a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_file_path, sep=sep, index_col=index_col)
        st.success("Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")
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

    # Predict on the test set and calculate precision and recall
    predictions = algo.test(testset)
    precision, recall = precision_recall_at_k(predictions)

    avg_precision = np.mean(list(precision.values()))
    avg_recall = np.mean(list(recall.values()))

    st.write(f"Average Precision: {avg_precision:.2f}")
    st.write(f"Average Recall: {avg_recall:.2f}")

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

    
# Train the model
algo = collaborative_filtering(ratings_df)

def find_similar_products_by_description(query, products, k=5):
    """Find similar products based on description."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-k:][::-1]
    similar_products = products.iloc[top_indices][['name', 'product_id']]
    similar_products['Relevance Score'] = cosine_sim[top_indices]
    similar_products.columns = ['Name', 'Product ID', 'Relevance Score']
    return similar_products


def personalized_recommendation(user_input, products, ratings, algo, user_name_to_id, k=5):
    """Generate personalized product recommendations."""
    user_id = user_name_to_id.get(user_input.lower())
    if user_id is not None:
        st.markdown(f"<p class='success-message'>Welcome back, {user_input.capitalize()}! Here are your personalized recommendations:</p>", unsafe_allow_html=True)
        recommended_products = hybrid_recommendation(user_id, products.copy(), ratings, algo, k)
        recommended_products = recommended_products[['name', 'product_id', 'hybrid_score']]
        recommended_products.columns = ['Name', 'Product ID', 'Relevance Score']
        st.table(recommended_products)
    else:
        st.markdown("<p class='warning-message'>User not found or you're a new user. Let's find some products for you.</p>", unsafe_allow_html=True)

def main_interaction_streamlit(products, ratings, algo, user_name_to_id):
    """Main interaction flow adapted for Streamlit."""
    user_input = st.text_input("👥 Enter your name for personalized recommendations or enter a product description below:", '')
    
    if user_input:
        personalized_recommendation(user_input, products, ratings, algo, user_name_to_id, 5)
    
    query = st.text_input("🔍 What are you looking for? Enter a product description or name:", '')
    
    if query:
        similar_products = find_similar_products_by_description(query, products, 5)
        if not similar_products.empty:
            st.markdown("<p class='success-message'>Top relevant products to your description input:</p>", unsafe_allow_html=True)
            st.table(similar_products)
        else:
            st.markdown("<p class='warning-message'>No similar products found.</p>", unsafe_allow_html=True)

if 'restart' not in st.session_state:
    st.session_state['restart'] = False

if not st.session_state['restart']:
    main_interaction_streamlit(products_df, ratings_df, algo, user_name_to_id)
