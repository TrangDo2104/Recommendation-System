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
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 18px;
            font-weight: bold;
            color: #666666;
        }
    </style>
""", unsafe_allow_html=True)


# Streamlit UI components
st.title("⭐ Welcom To Chimp AI's Recommendation System ⭐")

# Define your data loading and processing functions
def load_data(csv_file_path, sep=';', index_col=None):
    """Loads data from a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_file_path, sep=sep, index_col=index_col)
        # st.success("Data loaded successfully.")
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


# Calculate Similarity
def calculate_similarity(products_df, query=None):
    """Calculate TF-IDF cosine similarity based on product descriptions."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['description'])
    if query:
        query_vec = tfidf.transform([query])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Find Products by Description
def find_similar_products_by_description(query, products, k=5):
    """Find similar products based on a query description."""
    cosine_sim = calculate_similarity(products, query)
    top_indices = cosine_sim.argsort()[-k:][::-1]
    return products.iloc[top_indices]

# Recommendations Based on User Ratings
def recommend_for_user(user_id, ratings, products, k=5):
    """Recommend products based on a user's past high ratings."""
    user_ratings = ratings_df[ratings_df['user_name'] == user_id]
    high_rated_products = user_ratings[user_ratings['rating'] > 3.5]['Product ID'].unique()

    if len(high_rated_products) == 0:
        st.write("No high-rated products for user. Trying top-rated products...")
        # Optionally, recommend top-rated products here
        return pd.DataFrame()

    similar_products = pd.DataFrame()
    for product_id in high_rated_products:
        if product_id in products['Product ID'].values:
            product_desc = products.loc[products['Product ID'] == product_id, 'description'].iloc[0]
            sim_products = find_similar_products_by_description(product_desc, products, k)
            similar_products = pd.concat([similar_products, sim_products], axis=0).drop_duplicates().head(k)
    return similar_products

# UI for user input
user_input = st.text_input("Enter your user ID for personalized recommendations:")
if user_input:
    recommended_products = recommend_for_user(user_input, ratings_df, products_df, 5)
    if not recommended_products.empty:
        st.write(recommended_products)
    else:
        st.write("Unable to find recommendations based on user history.")

product_description_query = st.text_input("Or enter a product description to find similar products:")
if product_description_query:
    similar_products = find_similar_products_by_description(product_description_query, products_df, 5)
    if not similar_products.empty:
        st.write(similar_products)
    else:
        st.write("No similar products found based on the description.")
