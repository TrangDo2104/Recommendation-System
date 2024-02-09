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

products_df = load_data(product_metadata_path, sep=';')
ratings_df = load_data(user_ratings_path, sep=';', index_col='User')

products= products_df[['Product ID','Product Name', 'Product Price [SEK]','Product Description']]
products.columns = ['product_id', 'name', 'price', 'description']
products.head()

ratings = ratings_df.reset_index().melt(id_vars='User', var_name='Item', value_name='Rating')
ratings = ratings[ratings['Rating'] > 0]

# Convert 'user_name' to a categorical type and then to numerical codes
ratings['user_id'] = ratings['User'].astype('category').cat.codes
ratings.columns = ['user_name', 'product_id', 'rating', 'user_id']
user_name_to_id = pd.Series(ratings['user_id'].values, index=ratings['user_name'].str.lower()).to_dict()

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

# Hybrid recommendation logic
def hybrid_recommendation(user_id, product_name, products_df, ratings_df, algo, k=5):
    # For user-based recommendation
    if user_id is not None:
        cf_predictions = [algo.predict(user_id, pid).est for pid in products_df['product_id']]
        products_df['cf_score'] = cf_predictions
    else:
        products_df['cf_score'] = 0  # Default to 0 if no user_id

    # For product-based recommendation
    if product_name:
        cosine_sim = calculate_similarity(products_df)
        # Get the index of the product that matches the name
        idx = products_df.index[products_df['name'].str.lower() == product_name.lower()].tolist()
        if idx:
            sim_scores = list(enumerate(cosine_sim[idx[0]]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]  # Ignore the first one as it is the product itself
            product_indices = [i[0] for i in sim_scores]
            products_df['cbf_score'] = 0
            products_df.loc[product_indices, 'cbf_score'] = [score[1] for score in sim_scores]
        else:
            products_df['cbf_score'] = 0  # Default to 0 if no product_name match
    else:
        products_df['cbf_score'] = 0  # Default to 0 if no product_name

    # Hybrid score: average of CF and CBF scores
    products_df['hybrid_score'] = (products_df['cf_score'] + products_df['cbf_score']) / 2
    
    recommended_products = products_df.sort_values('hybrid_score', ascending=False).head(k)
    
    return recommended_products

# Streamlit inputs for user and product name
user_input = st.text_input("Enter username (for user-based recommendations):")
product_input = st.text_input("Enter product name (for product-based recommendations):")

# Train the model
algo = collaborative_filtering(ratings_df)

# Button for user-based recommendations
if st.button('Get User-based Recommendations'):
    if user_input:
        user_id = user_name_to_id.get(user_input.lower(), None)
        if user_id is not None:
            recommendations = hybrid_recommendation(user_id, None, products_df.copy(), ratings_df, algo)
            st.write(f"Recommendations for user: {user_input}")
            st.table(recommendations[['name', 'hybrid_score']])
        else:
            st.write("User not found. Please enter a valid username.")
    else:
        st.write("Please enter a username.")

# Button for product-based recommendations
if st.button('Get Product-based Recommendations'):
    if product_input:
        recommendations = hybrid_recommendation(None, product_input, products_df.copy(), ratings_df, algo)
        if not recommendations.empty:
            st.write(f"Products similar to {product_input}:")
            st.table(recommendations[['name', 'hybrid_score']])
        else:
            st.write(f"No similar products found for {product_input}.")
    else:
        st.write("Please enter a product name.")

if st.button('Train and Evaluate CF Model'):
    algo = collaborative_filtering(ratings_df)
