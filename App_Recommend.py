# 📄 File: App_Recommend.py

import streamlit as st
import pandas as pd
import pickle
import os
import difflib

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

if 'popularity_score' not in df.columns:
    df['popularity_score'] = (
        df['stars'] * 0.5 +
        df['reviews'] * 0.3 +
        df['boughtInLastMonth'] * 0.2
    )

if 'cluster' not in df.columns:
    df['cluster'] = kmeans.labels_

features = df[['stars', 'reviews', 'price', 'boughtInLastMonth']]
scaled_features = scaler.transform(features)

# ================================
# 🔍 SEARCH (MULTI MATCH)
# ================================

def search_products(product_name, n=5):
    
    titles = df['title'].fillna("").tolist()
    titles_lower = [t.lower() for t in titles]
    
    matches = difflib.get_close_matches(product_name.lower(), titles_lower, n=n, cutoff=0.4)
    
    result_df = pd.DataFrame()
    
    for match in matches:
        for i, t in enumerate(titles_lower):
            if t == match:
                result_df = pd.concat([result_df, df.iloc[[i]]])
    
    return result_df.drop_duplicates()

# ================================
# POPULARITY
# ================================

def popularity_recommend():
    return df.sort_values(by='popularity_score', ascending=False) \
            .head(5) \
            .to_dict(orient='records')

# ================================
# 🔥 ENSEMBLE MODEL
# ================================

def ensemble_recommend(product_title):
    
    matches = df[df['title'] == product_title]
    
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

st.title("🛒 Smart Product Recommendation Engine")

product_input = st.text_input("Enter Product Name")

selected_product = None

# Show suggestions
if product_input:
    matched_products = search_products(product_input)
    
    if not matched_products.empty:
        selected_product = st.selectbox(
            "Did you mean:",
            matched_products['title'].tolist()
        )
    else:
        st.warning("No similar products found")

# Recommendation button
if st.button("Recommend"):
    
    if selected_product:
        st.success(f"Showing results for: {selected_product}")
        results = ensemble_recommend(selected_product)
    else:
        st.warning("Showing popular products")
        results = popularity_recommend()
    
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