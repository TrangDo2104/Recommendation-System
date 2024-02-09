import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS for styling remains unchanged

# Streamlit UI components
st.title("⭐ Welcome To Chimp AI's Recommendation System ⭐")

# Define your data loading and processing functions
def load_data(csv_file_path, sep=';', index_col=None):
    """Loads data from a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_file_path, sep=sep, index_col=index_col)
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return None
        
# Load product metadata and user ratings data
product_metadata_path = 'Makeup_Products_Metadata.csv'  # Update path as necessary
user_ratings_path = 'User_review_data.csv'  # Update path as necessary

products_df = load_data(product_metadata_path, sep=';')
ratings_df = load_data(user_ratings_path, sep=';', index_col='User')

# Ensure DataFrame manipulations match your data structure
products_df = products_df[['Product ID','Product Name', 'Product Price [SEK]','Product Description']]
products_df.columns = ['product_id', 'name', 'price', 'description']

ratings_df = ratings_df.reset_index().melt(id_vars='User', var_name='Item', value_name='Rating')
ratings_df = ratings_df[ratings_df['Rating'] > 0]

# Convert 'User' to a categorical type and then to numerical codes
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
def find_similar_products_by_description(query, products_df, k=5):
    """Find similar products based on a query description."""
    cosine_sim = calculate_similarity(products_df, query)
    top_indices = cosine_sim.argsort()[-k:][::-1]
    return products_df.iloc[top_indices]

def get_global_top_rated_products(ratings_df, products_df, k=5):
    """Get top-rated products globally based on average rating >= 4."""
    # Ensure product_id is of the same data type in both DataFrames
    ratings_df['product_id'] = ratings_df['product_id'].astype(str)
    products_df['product_id'] = products_df['product_id'].astype(str)

    # Calculate average rating for each product
    avg_ratings = ratings_df.groupby('product_id').agg(
        avg_rating=('rating', 'mean'),
        count_ratings=('rating', lambda x: (x >= 4).sum())
    ).reset_index()

    # Filter for products with avg_rating >= 4 and sort by count_ratings
    top_rated_products = avg_ratings[avg_ratings['avg_rating'] >= 4].sort_values(by='count_ratings', ascending=False).head(k)

    # Join with products_df to get product details
    top_rated_details = pd.merge(top_rated_products, products_df, on='product_id', how='inner')[['product_id', 'name', 'avg_rating', 'count_ratings']]

    return top_rated_details

def recommend_for_user(user_input, ratings_df, products_df, k=5):
    """Recommend products based on a user's past highest-rated product."""
    user_input_lower = user_input.lower()
    user_ratings = ratings_df[ratings_df['user_name'].str.lower() == user_input_lower]

    if user_ratings.empty:
        st.write("No ratings found for this user. Showing global top-rated products instead.")
        return get_global_top_rated_products(ratings_df, products_df, k)

    # Filter ratings for products rated above 3.5
    high_rated = user_ratings[user_ratings['rating'] > 3.5].copy()
    if high_rated.empty:
        st.write("No high-rated products for this user. Showing global top-rated products instead.")
        return get_global_top_rated_products(ratings_df, products_df, k)

    # Find the highest-rated product
    highest_rated_product_id = high_rated.loc[high_rated['rating'].idxmax(), 'product_id']

    # Convert product_id to string for consistent data type
    highest_rated_product_id_str = str(highest_rated_product_id)
    products_df['product_id'] = products_df['product_id'].astype(str)

    # Find and exclude the highest-rated product from the recommendations if present
    highest_rated_product_desc = products_df.loc[products_df['product_id'] == highest_rated_product_id_str, 'description'].iloc[0]
    similar_products = find_similar_products_by_description(highest_rated_product_desc, products_df, k + 1)  # +1 to account for the product itself
    similar_products = similar_products[similar_products['product_id'] != highest_rated_product_id_str].head(k)

    return similar_products

# UI for user input
user_input = st.text_input("Enter your username for personalized recommendations:")
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
        st.write("No similar products found based")
