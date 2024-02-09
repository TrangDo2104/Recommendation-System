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

# Calculate similarity for content-based filtering
def calculate_similarity(products_df, query=None):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['description'])
    if query:
        query_vec = tfidf.transform([query])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def find_similar_products_by_description(query, products, k=5):
    """Find similar products based on description."""
    cosine_sim = calculate_similarity(products, query)
    top_indices = cosine_sim.argsort()[-k:][::-1]
    similar_products = products.iloc[top_indices][['name', 'product_id']]
    similar_products['Relevance Score'] = cosine_sim[top_indices]
    similar_products.columns = ['Name', 'Product ID', 'Relevance Score']
    return similar_products

# New function to recommend based on user history or top-rated products
def recommend_based_on_user_or_top_rated(user_input, products, ratings, user_name_to_id, k=5):
    user_id = user_name_to_id.get(user_input.lower())
    if user_id is not None:
        # Find products rated > 3.5 by the user
        high_rated_products = ratings[(ratings['user_name'].str.lower() == user_input.lower()) & (ratings['rating'] > 3.5)]
        if not high_rated_products.empty:
            # For each product, find similar ones based on description
            recommendations = pd.DataFrame()
            for product_id in high_rated_products['product_id'].unique():
                # Check if the product exists in the products DataFrame
                if not products[products['product_id'] == product_id].empty:
                    product_desc = products[products['product_id'] == product_id]['description'].iloc[0]
                    similar_products = find_similar_products_by_description(product_desc, products, k)
                    recommendations = pd.concat([recommendations, similar_products], ignore_index=True)
                else:
                    # If the product ID is not found in the products DataFrame, log or display an error
                    st.error(f"Product ID {product_id} not found in the database.")
            # After finding similar products for all high-rated products, deduplicate and sort
            recommendations = recommendations.drop_duplicates().sort_values('Relevance Score', ascending=False).head(k)
        else:
            # If no high-rated products, recommend top-rated globally
            top_rated_global = ratings.groupby('product_id')['rating'].mean().sort_values(ascending=False).head(k).index
            recommendations = products[products['product_id'].isin(top_rated_global)]
    else:
        st.markdown("<p class='warning-message'>User not found or you're a new user. Please search by product description.</p>", unsafe_allow_html=True)
        recommendations = pd.DataFrame()
    return recommendations

# Main interaction flow adapted for Streamlit with improved user interface
def main_interaction_streamlit(products, ratings, user_name_to_id):
    """Main interaction flow adapted for Streamlit."""
    # User input section remains unchanged

    # Adapted logic for personalized recommendations
    user_input = st.text_input("Enter your name to see recommendations based on your ratings or top-rated products.", key='user_input_name')
    if user_input:
        recommended_products = recommend_based_on_user_or_top_rated(user_input, products, ratings, user_name_to_id, 5)
        if not recommended_products.empty:
            st.table(recommended_products[['Name', 'Product ID', 'Relevance Score']])
        else:
            st.markdown("<p class='error-message'>No recommendations available.</p>", unsafe_allow_html=True)
            
    # New section for searching by product description
    st.markdown("### Search Products by Description")
    product_description_query = st.text_input("Enter product description to find similar products.", key='product_desc_search')
    if product_description_query:
        similar_products = find_similar_products_by_description(product_description_query, products, 5)
        if not similar_products.empty:
            st.table(similar_products[['Name', 'Product ID', 'Relevance Score']])
        else:
            st.markdown("<p class='error-message'>No similar products found based on the description.</p>", unsafe_allow_html=True)

    # Product description search functionality remains unchanged

if 'restart' not in st.session_state:
    st.session_state['restart'] = False

if not st.session_state['restart']:
    main_interaction_streamlit(products_df, ratings_df, user_name_to_id)
