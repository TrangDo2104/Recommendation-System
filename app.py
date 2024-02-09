import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inject custom CSS

# Custom CSS for styling
st.markdown("""
<style>
    /* Custom styling */
    /* Main content area */
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Input widgets styling */
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    
    /* Messages and alerts */
    .stAlert {
        border-radius: 20px;
    }
    
    .custom-message {
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
# Streamlit UI components
# Use st.markdown with HTML and CSS for center alignment
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

def find_users_without_high_rated_products(ratings_df, threshold=3.5):
    """Find users who do not have any products rated above the specified threshold."""
    # Group by user_name, filter those where max rating is <= threshold
    users_max_rating = ratings_df.groupby('user_name')['rating'].max()
    users_without_high_ratings = users_max_rating[users_max_rating <= threshold].index.tolist()

    return users_without_high_ratings

# Assuming 'ratings_df' is already defined and contains the columns 'user_name' and 'rating'
users_without_high_ratings = find_users_without_high_rated_products(ratings_df)
# Display Message Function
def display_message(message, category="info"):
    if category == "info":
        st.markdown(f"<div class='custom-message info-message'>{message}</div>", unsafe_allow_html=True)
    elif category == "warning":
        st.markdown(f"<div class='custom-message warning-message'>{message}</div>", unsafe_allow_html=True)
    elif category == "error":
        st.markdown(f"<div class='custom-message error-message'>{message}</div>", unsafe_allow_html=True)
    elif category == "success":
        st.markdown(f"<div class='custom-message success-message'>{message}</div>", unsafe_allow_html=True)

# UI for personalized recommendations
st.markdown("### Uncover Your Personalized Treasures")
st.markdown("##### Looking for products tailored just for you?")
user_input = st.text_input("Enter your username:", placeholder="Username")
if user_input:
    # Assuming recommend_for_user returns DataFrame
    recommended_products = recommend_for_user(user_input, ratings_df, products_df, 5)
    if not recommended_products.empty:
        st.markdown(f"<div class='custom-message'>Welcome {user_input}, here are your personalized products:</div>", unsafe_allow_html=True)
        st.dataframe(recommended_products)  # Use st.dataframe for better control over the display
    else:
        st.warning(f"Couldn't find specific recommendations for {user_input}. Here are some top-rated products instead.")
        global_top_products = get_global_top_rated_products(ratings_df, products_df, 5)
        st.dataframe(global_top_products)

st.markdown("---")  # Separator

# UI for finding products by description
st.markdown("### Explore Products Tailored to Your Taste")
st.markdown("##### Looking for something specific?")
product_description_query = st.text_input("Enter the product description or name here:", placeholder="Product description")
if product_description_query:
    similar_products = find_similar_products_by_description(product_description_query, products_df, 5)
    if not similar_products.empty:
        st.markdown("<div class='custom-message'>Found some products that might interest you:</div>", unsafe_allow_html=True)
        st.dataframe(similar_products)
    else:
        st.error("No matches found based on the description. Try different keywords!")

# # Now, for displaying messages in a more interactive and interesting way:
# def display_message(message, category="info"):
#     if category == "info":
#         st.markdown(f"<p style='color: #4F8BF9; font-size: 20px;'>{message}</p>", unsafe_allow_html=True)
#     elif category == "warning":
#         st.markdown(f"<p style='color: orange; font-size: 20px;'>{message}</p>", unsafe_allow_html=True)
#     elif category == "error":
#         st.markdown(f"<p style='color: red; font-size: 20px;'>{message}</p>", unsafe_allow_html=True)
#     elif category == "success":
#         st.markdown(f"<p style='color: green; font-size: 20px;'>{message}</p>", unsafe_allow_html=True)

# # Example usage of display_message:
# if user_input:
#     recommended_products = recommend_for_user(user_input, ratings_df, products_df, 5)
#     if not recommended_products.empty:
#         display_message("Here are your personalized recommendations:", "success")
#         st.table(recommended_products)
#     else:
#         display_message("Unable to find recommendations based on your history. But don't worry, here are some top-rated products just for you!", "warning")
#         global_top_products = get_global_top_rated_products(ratings_df, products_df, 5)
#         st.table(global_top_products)
# else:
#     display_message("Enter your username to see personalized recommendations, or explore products by description!", "info")

# if product_description_query:
#     similar_products = find_similar_products_by_description(product_description_query, products_df, 5)
#     if not similar_products.empty:
#         display_message("Found some products that might interest you:", "success")
#         st.table(similar_products)
#     else:
#         display_message("No similar products found based on the description.", "error")
