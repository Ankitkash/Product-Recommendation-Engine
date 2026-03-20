# 🛒 Product Recommendation Engine

## 🚀 Overview
The Product Recommendation Engine is a machine learning-based system designed to recommend similar and relevant products based on product attributes such as rating, price, reviews, and demand.

This project implements a **hybrid recommendation system** combining multiple machine learning techniques to improve recommendation quality and handle different scenarios like cold-start and similarity-based suggestions.

---

## 🎯 Objectives
- Recommend similar products based on user input
- Identify trending and popular products
- Build a scalable and explainable recommendation system
- Deploy the model using a web interface

---

## 🧠 Algorithms Used

### 1. K-Nearest Neighbors (KNN)
- Finds similar products using feature similarity
- Uses Euclidean distance on scaled features

### 2. K-Means Clustering
- Groups similar products into clusters
- Recommends products from the same cluster

### 3. Popularity-Based Filtering
- Recommends top products based on:
  - Ratings
  - Number of reviews
  - Recent demand

### 4. Hybrid Recommendation System
- Combines KNN and popularity score
- Improves accuracy and relevance

---

## 📊 Dataset
- Large-scale Amazon product dataset (1.4M+ rows)
- Features used:
  - Product Title
  - Rating (stars)
  - Number of Reviews
  - Price
  - Purchase Frequency (boughtInLastMonth)
  - Product Image URL

---

## ⚙️ Project Workflow

1. Data Cleaning
   - Handle missing values
   - Remove duplicates
   - Sample dataset for performance

2. Feature Engineering
   - Select numerical features
   - Apply MinMax Scaling

3. Model Building
   - Train KNN model
   - Train KMeans clustering model
   - Create popularity score

4. Model Saving
   - Save models using pickle

5. Deployment
   - Build UI using Streamlit
   - Provide real-time recommendations

---

## 🖥️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

---
