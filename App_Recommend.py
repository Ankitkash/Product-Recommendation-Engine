import streamlit as st
import pandas as pd
import pickle
import os

# ================================
# LOAD FILES
# ================================

BASE_DIR = os.path.dirname(__file__)

knn_model = pickle.load(open(os.path.join(BASE_DIR, "knn_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
kmeans = pickle.load(open(os.path.join(BASE_DIR, "kmeans_model.pkl"), "rb"))
df = pickle.load(open(os.path.join(BASE_DIR, "data.pkl"), "rb"))

# ================================
# DATA PREPROCESSING (FIXES)
# ================================

# Reset index to avoid mismatch issues
df = df.reset_index(drop=True)

# Ensure required columns exist
required_cols = ['stars', 'reviews', 'price', 'boughtInLastMonth']
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")

# Create popularity_score if not exists
if 'popularity_score' not in df.columns:
    df['popularity_score'] = (
        df['stars'] * 0.5 +
        df['reviews'] * 0.3 +
        df['boughtInLastMonth'] * 0.2
    )

# Create cluster column if not exists
if 'cluster' not in df.columns:
    df['cluster'] = kmeans.labels_

# Feature scaling
features = df[['stars', 'reviews', 'price', 'boughtInLastMonth']]
scaled_features = scaler.transform(features)

# ================================
# FUNCTIONS
# ================================

def search_product(product_name):
    return df[df['title'].str.lower().str.contains(product_name.lower(), na=False)]

def knn_recommend(product_name):
    matches = search_product(product_name)
    
    if matches.empty:
        return []
    
    idx = matches.index[0]
    
    distances, indices = knn_model.kneighbors([scaled_features[idx]])
    
    return df.iloc[indices[0][1:]]

def kmeans_recommend(product_name):
    matches = search_product(product_name)
    
    if matches.empty:
        return []
    
    cluster = matches.iloc[0]['cluster']
    
    return df[df['cluster'] == cluster] \
            .sort_values(by='stars', ascending=False) \
            .head(5)

def popularity_recommend():
    return df.sort_values(by='popularity_score', ascending=False).head(5)

def hybrid_recommend(product_name):
    matches = search_product(product_name)
    
    if matches.empty:
        return []
    
    idx = matches.index[0]
    
    distances, indices = knn_model.kneighbors([scaled_features[idx]])
    
    results = []
    
    for i in indices[0][1:]:
        row = df.iloc[i]
        score = row['popularity_score']
        results.append((row, score))
    
    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    return [r[0] for r in results]

# ================================
# UI
# ================================

st.set_page_config(page_title="Product Recommender", layout="wide")

st.title("🛒 Product Recommendation Engine")

option = st.selectbox(
    "Choose Recommendation Type",
    ["Hybrid (Best)", "KNN", "KMeans", "Popularity"]
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
    
    if len(results) == 0:
        st.warning("❌ No products found")
    
    else:
        for item in results:
            
            # Safe image display
            if 'imgUrl' in item and pd.notna(item['imgUrl']):
                try:
                    st.image(item['imgUrl'], width=150)
                except:
                    st.write("🖼️ Image not available")
            
            st.subheader(item['title'])
            
            st.write(f"⭐ Rating: {item['stars']}")
            st.write(f"💰 Price: ₹{item['price']}")
            
            if 'reviews' in item:
                st.write(f"📝 Reviews: {item['reviews']}")
            
            st.write("---")