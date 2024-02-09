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

# Function to calculate similarity for content-based filtering
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

def recommend_based_on_user_or_top_rated(user_input, products, ratings, user_name_to_id, k=5):
    """Generate recommendations based on user's past high ratings or top-rated products."""
    user_id = user_name_to_id.get(user_input.lower())
    recommendations = pd.DataFrame()
    if user_id is not None:
        high_rated_products = ratings[(ratings['user_name'].str.lower() == user_input.lower()) & (ratings['rating'] > 3.5)]
        if not high_rated_products.empty:
            for product_id in high_rated_products['product_id'].unique():
                if not products[products['product_id'] == product_id].empty:
                    product_desc = products[products['product_id'] == product_id]['description'].iloc[0]
                    similar_products = find_similar_products_by_description(product_desc, products, k)
                    recommendations = pd.concat([recommendations, similar_products], ignore_index=True)
            if not recommendations.empty:
                recommendations = recommendations.drop_duplicates().sort_values('Relevance Score', ascending=False).head(k)
            else:
                st.error("No similar products found based on user's ratings.")
        else:
            st.markdown("<p class='warning-message'>No high-rated products found for this user. Showing top-rated products instead.</p>", unsafe_allow_html=True)
            # Implement logic for recommending top-rated products if needed
    else:
        st.markdown("<p class='warning-message'>User not found. Please search by product description.</p>", unsafe_allow_html=True)
    return recommendations

# Main interaction flow
def main_interaction_streamlit(products, ratings, user_name_to_id):
    user_input = st.text_input("Enter your name to see recommendations:", key='user_input_name')
    if user_input:
        recommended_products = recommend_based_on_user_or_top_rated(user_input, products, ratings, user_name_to_id, 5)
        if not recommended_products.empty:
            st.table(recommended_products)
        else:
            st.markdown("<p class='error-message'>Unable to generate recommendations.</p>", unsafe_allow_html=True)
    
    product_description_query = st.text_input("Or enter a product description to find similar products:", key='product_desc_search')
    if product_description_query:
        similar_products = find_similar_products_by_description(product_description_query, products, 5)
        if not similar_products.empty:
            st.table(similar_products)
        else:
            st.markdown("<p class='error-message'>No similar products found based on the description.</p>", unsafe_allow_html=True)

# Convert 'user_name' to a categorical type and then to numerical codes (adjust according to your DataFrame)
ratings_df['user_id'] = ratings_df['user_name'].astype('category').cat.codes
user_name_to_id = pd.Series(ratings_df['user_id'].values, index=ratings_df['user_name'].str.lower()).to_dict()

if __name__ == '__main__':
    main_interaction_streamlit(products_df, ratings_df, user_name_to_id)
