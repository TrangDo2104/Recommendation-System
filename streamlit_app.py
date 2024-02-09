import streamlit as st
import itertools
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
st.title("⭐ Welcome To Chimp AI's Recommendation System ⭐")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from collections import defaultdict

# Define a function to load data safely
@st.cache_data
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

# Preprocessing product data
products_df = products_df[['Product ID', 'Product Name', 'Product Description']]
products_df.columns = ['product_id', 'name', 'description']

# Preprocessing ratings data according to the new specification
ratings_df = ratings_df.reset_index().melt(id_vars='User', var_name='Item', value_name='Rating')
ratings_df = ratings_df[ratings_df['Rating'] > 0]
ratings_df['user_id'] = ratings_df['User'].astype('category').cat.codes
ratings_df = ratings_df.rename(columns={'User': 'user_name', 'Item': 'product_id', 'Rating': 'rating'})
user_name_to_id = pd.Series(ratings_df['user_id'].values, index=ratings_df['user_name'].str.lower()).to_dict()

# Collaborative Filtering with Username
def collaborative_filtering_with_username(ratings_df):
    unique_users = ratings_df['user_name'].unique()
    user_ids = {user: i for i, user in enumerate(unique_users)}
    ratings_df['user_id'] = ratings_df['user_name'].apply(lambda x: user_ids[x])

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.04]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)

    algo = gs.best_estimator['rmse']
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    return algo, user_ids

algo, user_name_to_id = collaborative_filtering_with_username(ratings_df)

# Precision and Recall at K
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

def find_similar_products_by_description(query, products_df, k=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['description'])
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
    
    if isinstance(cosine_sim, np.ndarray):
        cosine_sim = cosine_sim.flatten()
        top_indices = cosine_sim.argsort()[-k:][::-1]
        return products_df.iloc[top_indices][['name', 'product_id']]
    else:
        st.error("Error computing cosine similarity.")
        return pd.DataFrame()

# Hybrid Recommendation
def hybrid_recommendation(username, k=5):
    user_id = user_name_to_id.get(username.lower())
    if user_id is None:
        return pd.DataFrame()

    user_ratings = ratings_df[ratings_df['user_name'].str.lower() == username.lower()]
    user_ratings = user_ratings.set_index('product_id')['rating'].to_dict()

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['description'])
    product_ids = products_df['product_id'].tolist()

    # CBF scores
    cbf_scores = np.zeros(len(products_df))
    similarity_matrix = cosine_similarity(tfidf_matrix)
    for pid, rating in user_ratings.items():
        if pid in product_ids:
            idx = product_ids.index(pid)
            similarities = similarity_matrix[idx]
            for i, sim in enumerate(similarities):
                if rating >= 3.5:
                    cbf_scores[i] += sim * 0.5
                else:
                    cbf_scores[i] -= sim * 0.5

    # CF scores
    cf_scores = np.array([algo.predict(user_id, pid).est for pid in product_ids])
    
    # Hybrid scores
    hybrid_scores = cf_scores * 0.5 + cbf_scores * 0.5
    products_df['hybrid_score'] = hybrid_scores

    return products_df.sort_values(by='hybrid_score', ascending=False).head(k)

if not products_df.empty and not ratings_df.empty:
    # Main interaction function adjusted for Streamlit
    # Initialize a counter for generating unique keys
    # Add this line before the main_interaction_streamlit function
    widget_counter = itertools.count()
    
    # Main interaction function adjusted for Streamlit
    def main_interaction_streamlit(products_df, ratings_df, user_name_to_id):
        user_input_key = next(widget_counter)
        user_query_key = next(widget_counter)
    
        user_input = st.text_input("Enter your name for personalized recommendations or explore as a guest:", key=user_input_key)
        query = st.text_input("Looking for something specific? Enter keywords to find related products:", key=user_query_key)
    
        if user_input:
            if user_input.lower() != 'guest':
                st.write(f"Welcome back, {user_input.capitalize()}! Here are your personalized recommendations:")
                recommended_products = recommended_products = hybrid_recommendation(user_input)
                st.dataframe(recommended_products[['name', 'product_id', 'hybrid_score']])
            else:
                st.write("Explore our products as a guest.")
    
        if query:
            similar_products = find_similar_products_by_description(query, products_df)
            st.write("Top relevant products to your description input:")
            st.dataframe(similar_products)


    if __name__ == "__main__":
        main_interaction_streamlit(products_df, ratings_df, user_name_to_id)
else:
    st.error("Failed to load data. Please check the file paths and try again.")

if __name__ == "__main__":
    # Ensure collaborative_filtering_with_username and other necessary functions are defined and called correctly
    main_interaction_streamlit(products_df, ratings_df, user_name_to_id)
