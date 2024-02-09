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

    # Calculate average rating for each product and count of ratings >= 4
    avg_ratings = ratings_df.groupby('product_id').agg(
        avg_rating=('rating', 'mean'),
        count_ratings=('rating', lambda x: (x >= 4).sum())
    ).reset_index()

    # Filter products with avg_rating >= 4, sort by count_ratings descending
    top_rated_products = avg_ratings[avg_ratings['avg_rating'] >= 4].sort_values(by='count_ratings', ascending=False).head(k)

    # Join with products_df to get product details
    top_rated_details = pd.merge(top_rated_products, products_df, on='product_id', how='left')

    return top_rated_details

def recommend_for_user(user_input, ratings_df, products_df, k=5):
    """Recommend products based on a user's past high ratings or global top-rated products."""
    # Attempt to find high-rated products for the user
    user_ratings = ratings_df[ratings_df['user_name'].str.lower() == user_input.lower()]
    
    high_rated_products = user_ratings[user_ratings['rating'] > 3.5]['product_id'].unique()

    similar_products = pd.DataFrame()
    if len(high_rated_products) > 0:
        for product_id in high_rated_products:
            if product_id in products_df['product_id'].values:
                product_desc = products_df.loc[products_df['product_id'] == product_id, 'description'].iloc[0]
                sim_products = find_similar_products_by_description(product_desc, products_df, k)
                similar_products = pd.concat([similar_products, sim_products], axis=0).drop_duplicates().head(k)
    else:
        # User has no high-rated products or not found, use global top-rated products instead
        similar_products = get_global_top_rated_products(ratings_df, products_df, k)

    if similar_products.empty:
        st.write("Unable to find recommendations based on user history. Showing global top-rated products instead.")
        similar_products = get_global_top_rated_products(ratings_df, products_df, k)
    
    return similar_products

# UI for user input
user_input = st.text_input("Enter your user name for personalized recommendations:")
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
