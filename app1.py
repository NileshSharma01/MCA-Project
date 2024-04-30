import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the data from the XLSX file
data = pd.read_excel('Newprocessed.xlsx')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# Apply preprocessing to recipe names
data['RecipeName'] = data['RecipeName'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(data['RecipeName'])

# Streamlit app
def main():
    st.title('Recipe Recommendation System')

    # Get user input
    recipe_name = st.text_input('Enter the name of the recipe:')

    if st.button('Recommend'):
        if recipe_name:
            # Preprocess input recipe name
            recipe_tokens = preprocess_text(recipe_name).split()

            # Compute TF-IDF matrix for input tokens
            input_tfidf_matrix = vectorizer.transform([recipe_name])

            # Compute cosine similarity
            similarities = cosine_similarity(input_tfidf_matrix, tfidf_matrix)

            # Get indices of top N similar recipes
            top_n_indices = np.argsort(similarities, axis=1)[:, -10:][:, ::-1]

            # Get the recipe names and cuisines for the top N similar recipes
            top_recipes = [(data.iloc[idx]['RecipeName'], data.iloc[idx]['Cuisine']) for idx in top_n_indices[0]]

            # Display recommendations
            st.subheader(f"Top 10 Recommendations for '{recipe_name}':")
            for i, recipe in enumerate(top_recipes, 1):
                st.write(f"{i}. Recipe: {recipe[0]}, Cuisine: {recipe[1]}")
        else:
            st.warning("Please enter a recipe name.")

if __name__ == '__main__':
    main()
