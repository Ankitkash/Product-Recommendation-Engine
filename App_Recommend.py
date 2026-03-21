# ================================
# PRODUCT RECOMMENDATION ENGINE
# ================================

import streamlit as st
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from difflib import get_close_matches

# ================================
# LOAD DATA
# ================================

@st.cache_data
def load_data():
    df = pd.read_csv("small_data.csv")
    
    # Cleaning
    df = df.dropna(subset=['title'])
    df = df.drop_duplicates(subset=['asin'])
    
    df['stars'] = df['stars'].fillna(df['stars'].mean())
    df['reviews'] = df['reviews'].fillna(0)
    df['price'] = df['price'].fillna(df['price'].mean())
    df['boughtInLastMonth'] = df['boughtInLastMonth'].fillna(0)
    
    # Fix image column
    df['imgUrl'] = df.get('imgUrl', "")
    df['imgUrl'] = df['imgUrl'].fillna("")
    
    return df

df = load_data()

# ================================
# FEATURE ENGINEERING
# ================================

features = df[['stars', 'reviews', 'price', 'boughtInLastMonth']]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# ================================
# MODELS
# ================================

knn_model = NearestNeighbors(n_neighbors=6)
knn_model.fit(scaled_features)

kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Popularity Score
df['popularity_score'] = (
    df['stars'] * 0.5 +
    df['reviews'] * 0.3 +
    df['boughtInLastMonth'] * 0.2
)

# ================================
# BEST MATCH FUNCTION (IMPORTANT)
# ================================

def find_best_match(product_name):
    titles = df['title'].str.lower().tolist()
    
    matches = get_close_matches(product_name.lower(), titles, n=1, cutoff=0.3)
    
    if not matches:
        return None
    
    best_match = matches[0]
    
    return df[df['title'].str.lower() == best_match].index[0]

# ================================
# RECOMMENDATION FUNCTIONS
# ================================

def knn_recommend(product_name):
    idx = find_best_match(product_name)
    
    if idx is None:
        return pd.DataFrame()
    
    distances, indices = knn_model.kneighbors([scaled_features[idx]])
    
    return df.iloc[indices[0][1:]]


def kmeans_recommend(product_name):
    idx = find_best_match(product_name)
    
    if idx is None:
        return pd.DataFrame()
    
    cluster = df.iloc[idx]['cluster']
    
    return df[df['cluster'] == cluster].sort_values(by='stars', ascending=False).head(5)


def popularity_recommend():
    return df.sort_values(by='popularity_score', ascending=False).head(5)


def hybrid_recommend(product_name):
    idx = find_best_match(product_name)
    
    if idx is None:
        return pd.DataFrame()
    
    distances, indices = knn_model.kneighbors([scaled_features[idx]])
    
    results = df.iloc[indices[0][1:]].copy()
    
    return results.sort_values(by='popularity_score', ascending=False)

# ================================
# UI
# ================================

st.title("🛒 Product Recommendation Engine")

option = st.selectbox(
    "Choose Recommendation Type",
    ["Hybrid", "KNN", "KMeans", "Popularity"]
)

product = st.text_input("Enter Product Name")

if st.button("Recommend"):
    
    if option == "KNN":
        results = knn_recommend(product)
        
    elif option == "KMeans":
        results = kmeans_recommend(product)
        
    elif option == "Popularity":
        results = popularity_recommend()
        
    else:
        results = hybrid_recommend(product)
    
    if results is None or results.empty:
        st.write("❌ No products found")
    
    else:
        for _, item in results.iterrows():
            
            # Image handling
            img = item.get('imgUrl', "")
            
            if img:
                st.image(img, width=150)
            else:
                st.write("No Image Available")
            
            st.subheader(item['title'])
            st.write(f"⭐ {item['stars']} | ₹{item['price']}")
            st.write("---")