# ================================
# STREAMLIT APP
# ================================

import streamlit as st
import pandas as pd
import pickle

# ================================
# LOAD FILES
# ================================

import os

BASE_DIR = os.path.dirname(__file__)

knn_model = pickle.load(open(os.path.join(BASE_DIR, "knn_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
kmeans = pickle.load(open(os.path.join(BASE_DIR, "kmeans_model.pkl"), "rb"))
df = pickle.load(open(os.path.join(BASE_DIR, "data.pkl"), "rb"))

features = df[['stars', 'reviews', 'price', 'boughtInLastMonth']]
scaled_features = scaler.transform(features)

# ================================
# FUNCTIONS
# ================================

def knn_recommend(product_name):
    matches = df[df['title'].str.contains(product_name.lower())]
    
    if matches.empty:
        return []
    
    idx = matches.index[0]
    distances, indices = knn_model.kneighbors([scaled_features[idx]])
    
    return df.iloc[indices[0][1:]]


def kmeans_recommend(product_name):
    matches = df[df['title'].str.contains(product_name.lower())]
    
    if matches.empty:
        return []
    
    cluster = matches.iloc[0]['cluster']
    
    return df[df['cluster'] == cluster].sort_values(by='stars', ascending=False).head(5)


def popularity_recommend():
    return df.sort_values(by='popularity_score', ascending=False).head(5)


def hybrid_recommend(product_name):
    matches = df[df['title'].str.contains(product_name.lower())]
    
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
        st.write("❌ No products found")
    
    else:
        for item in results:
            st.image(item['imgUrl'], width=150)
            st.write(item['title'])
            st.write(f"⭐ {item['stars']} | ₹{item['price']}")
            st.write("---")
