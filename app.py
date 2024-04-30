import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

# Load the trained model and vectorizer
vectorizer = joblib.load('models/vectorizer.joblib')
model = joblib.load('models/recommend_recipes.joblib')

# Function for preprocessing text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# Function to recommend recipes
def recommend_recipes(recipe_name):
    recipe_tokens = preprocess_text(recipe_name).split()

    # Compute TF-IDF matrix for input tokens
    input_tfidf_matrix = vectorizer.transform([recipe_name])

    # Compute cosine similarity
    similarities = cosine_similarity(input_tfidf_matrix, vectorizer.transform(data['RecipeName']))

    # Get indices of top N similar recipes
    top_n_indices = np.argsort(similarities, axis=1)[:, -10:][:, ::-1]

    # Get the recipe names and cuisines for the top N similar recipes
    top_recipes = [(data.iloc[idx]['RecipeName'], data.iloc[idx]['Cuisine']) for idx in top_n_indices[0]]

    return top_recipes

# Streamlit app
def main():
    st.title('Regional Food Recommendation System')

    # Get user input
    recipe_name = st.text_input('Enter the name of the recipe:')

    if st.button('Recommend'):
        if recipe_name:
            recommended_recipes = recommend_recipes(recipe_name)
            if recommended_recipes:
                st.subheader(f"Top 10 Recommendations for '{recipe_name}':")
                for i, recipe in enumerate(recommended_recipes, 1):
                    st.write(f"{i}. Recipe: {recipe[0]}, Cuisine: {recipe[1]}")
            else:
                st.write("No recommendations found.")
        else:
            st.warning("Please enter a recipe name.")

if __name__ == '__main__':
    main()
