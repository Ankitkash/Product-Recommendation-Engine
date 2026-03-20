import pickle
import numpy as np
import streamlit as st

# ── Load from Pickle ──────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("Recommendation_System.pkl", "rb") as f:
        return pickle.load(f)

data   = load_models()
df     = data["df"]
scaled = data["scaled"]
knn    = data["knn"]

# ── Helpers ───────────────────────────────────────────────────
def find(name):
    m = df[df["title"].str.contains(name.strip(), case=False, na=False)]
    return None if m.empty else m.loc[m["stars"].idxmax()]

def get_neighbors(product):
    dists, idxs = knn.kneighbors([scaled[product.name]])
    return dists[0][1:], idxs[0][1:]

# ── Recommenders ──────────────────────────────────────────────
def knn_recommend(name):
    p = find(name)
    if p is None: return []
    _, idxs = get_neighbors(p)
    return df.iloc[idxs].to_dict("records")

def kmeans_recommend(name):
    p = find(name)
    if p is None: return []
    same = df[(df["cluster"] == p["cluster"]) & (df.index != p.name)]
    return same.nlargest(5, "stars").to_dict("records")

def popularity_recommend():
    return df.nlargest(5, "pop_score").to_dict("records")

def hybrid_recommend(name):
    p = find(name)
    if p is None: return []
    dists, idxs = get_neighbors(p)
    rows = df.iloc[idxs]
    scores = 0.7 * (1 / (1 + dists)) + 0.3 * rows["pop_score"].values
    return rows.iloc[np.argsort(-scores)].to_dict("records")

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Amazon Recommender", page_icon="🛒")
st.title("🛒 Amazon Recommendation System")

method = st.selectbox("Method", ["Hybrid", "KNN", "KMeans", "Popularity"])
query  = st.text_input("Product name", placeholder="e.g. wireless headphones")

if st.button("Recommend"):
    if method != "Popularity" and not query.strip():
        st.warning("Please enter a product name.")
    else:
        fn = {"KNN": knn_recommend, "KMeans": kmeans_recommend,
              "Popularity": lambda _: popularity_recommend(), "Hybrid": hybrid_recommend}
        results = fn[method](query)

        if not results:
            st.error(f"No products found for '{query}'.")
        else:
            for col, item in zip(st.columns(len(results)), results):
                with col:
                    img = item.get("imgUrl", "")
                    st.image(img if isinstance(img, str) and img.startswith("http")
                             else "https://via.placeholder.com/140", width=130)
                    st.markdown(f"**{item['title'][:60]}...**")
                    st.caption(f"⭐ {item['stars']} | 💰 ₹{item['price']:.0f} | 💬 {int(item['reviews']):,}")