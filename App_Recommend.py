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
# DATA PREPROCESSING
# ================================

df = df.reset_index(drop=True)

# Create missing columns safely
if 'popularity_score' not in df.columns:
    df['popularity_score'] = (
        df['stars'] * 0.5 +
        df['reviews'] * 0.3 +
        df['boughtInLastMonth'] * 0.2
    )

if 'cluster' not in df.columns:
    df['cluster'] = kmeans.labels_

# Feature scaling
features = df[['stars', 'reviews', 'price', 'boughtInLastMonth']]
scaled_features = scaler.transform(features)

# ================================
# HELPER FUNCTION
# ================================

def search_product(product_name):
    return df[df['title'].str.lower().str.contains(product_name.lower(), na=False)]

# ================================
# INDIVIDUAL MODELS
# ================================

def knn_recommend(product_name):
    matches = search_product(product_name)
    if matches.empty:
        return []
    
    idx = matches.index[0]
    distances, indices = knn_model.kneighbors([scaled_features[idx]])
    
    return df.iloc[indices[0][1:]].to_dict(orient='records')


def kmeans_recommend(product_name):
    matches = search_product(product_name)
    if matches.empty:
        return []
    
    cluster = matches.iloc[0]['cluster']
    
    return df[df['cluster'] == cluster] \
            .sort_values(by='stars', ascending=False) \
            .head(5) \
            .to_dict(orient='records')


def popularity_recommend():
    return df.sort_values(by='popularity_score', ascending=False) \
            .head(5) \
            .to_dict(orient='records')


# ================================
# ENSEMBLE MODEL (BEST)
# ================================

def ensemble_recommend(product_name):
    matches = search_product(product_name)
    
    # fallback if no product
    if matches.empty:
        return popularity_recommend()
    
    idx = matches.index[0]
    final_results = {}

    # ---------- KNN ----------
    try:
        distances, indices = knn_model.kneighbors([scaled_features[idx]])
        
        for i in indices[0][1:]:
            item = df.iloc[i].to_dict()
            key = item['title']
            
            if key not in final_results:
                final_results[key] = item
                final_results[key]['score'] = 0
            
            final_results[key]['score'] += 3
    except:
        pass

    # ---------- KMeans ----------
    try:
        cluster = matches.iloc[0]['cluster']
        cluster_items = df[df['cluster'] == cluster].head(10)
        
        for _, row in cluster_items.iterrows():
            item = row.to_dict()
            key = item['title']
            
            if key not in final_results:
                final_results[key] = item
                final_results[key]['score'] = 0
            
            final_results[key]['score'] += 2
    except:
        pass

    # ---------- Popularity ----------
    try:
        pop_items = popularity_recommend()
        
        for item in pop_items:
            key = item['title']
            
            if key not in final_results:
                final_results[key] = item
                final_results[key]['score'] = 0
            
            final_results[key]['score'] += 1
    except:
        pass

    # ---------- FINAL SORT ----------
    results = list(final_results.values())
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    return results[:5]


# ================================
# UI
# ================================

st.set_page_config(page_title="Product Recommender", layout="wide")

st.title("🛒 Product Recommendation Engine (Ensemble Model)")

product = st.text_input("Enter Product Name")

if st.button("Recommend"):
    
    results = ensemble_recommend(product)
    
    if len(results) == 0:
        st.warning("❌ No products found")
    
    else:
        for item in results:
            
            img = item.get('imgUrl')
            if img:
                try:
                    st.image(img, width=150)
                except:
                    st.write("🖼️ Image not available")
            
            st.subheader(item.get('title', 'No Title'))
            st.write(f"⭐ Rating: {item.get('stars', 'N/A')}")
            st.write(f"💰 Price: ₹{item.get('price', 'N/A')}")
            st.write(f"🔥 Score: {item.get('score', 0)}")
            st.write("---")