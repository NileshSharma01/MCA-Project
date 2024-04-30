import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

# Load the data from the XLSX file
data = pd.read_excel('Newprocessed.xlsx')

# Preprocessing steps
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numerical values
    text = re.sub(r'\d+', '', text)
    
    # Remove leading/trailing white spaces
    text = text.strip()
    
    return text

# Apply preprocessing to recipe names
data['RecipeName'] = data['RecipeName'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(data['RecipeName'])

def get_recommendations(recipe_tokens, data, tfidf_matrix, top_n=10):
    # Check if any input token is not present in the dataset
    for token in recipe_tokens:
        if token not in vectorizer.vocabulary_:
            print(f"Word '{token}' not found. Halting process.")
            return [], []

    # Compute the TF-IDF matrix for the input tokens
    input_tfidf_matrix = vectorizer.transform(recipe_tokens)

    # Compute the cosine similarity between the input tokens and all recipe names
    similarities = cosine_similarity(input_tfidf_matrix, tfidf_matrix)

    # Get indices of top N similar recipes
    top_recipe_indices = np.argsort(similarities, axis=1)[:, -top_n:][:, ::-1]

    # Get the recipe names and cuisines for the top N similar recipes
    top_recipes = []
    for indices in top_recipe_indices:
        recipes = [(data.iloc[row]['RecipeName'], data.iloc[row]['Cuisine']) for row in indices]
        top_recipes.append(recipes)

    # Compute the average cosine similarity score for each recipe
    avg_similarities = np.mean(similarities, axis=0)

    # Sort the recipes based on the average cosine similarity score
    sorted_recipes = sorted(zip(avg_similarities, data['RecipeName'], data['Cuisine']), reverse=True)[:top_n]

    return top_recipes, sorted_recipes

# Function to prompt user for input recipe name and get recommendations
def recommend_recipes():
    recipe_name = input("Enter the name of the recipe: ")
    recipe_tokens = recipe_name.split()

    token_recommendations, avg_recommendations = get_recommendations(recipe_tokens, data, tfidf_matrix, top_n=10)

    if token_recommendations:
        print(f"\nRecommendations for '{recipe_name}' (Token-wise):")
        for i, recipes in enumerate(token_recommendations, 1):
            print(f"Recommendations for token {i}:")
            print(tabulate(recipes, headers=['RecipeName', 'Cuisine'], tablefmt='grid'))

    if avg_recommendations:
        print("\n")
        print(f"Recommendations for '{recipe_name}' (Average):")
        print(tabulate(avg_recommendations, headers=['Average Similarity','RecipeName', 'Cuisine'], tablefmt='grid'))

# Train the model and save it
vectorizer_file = 'vectorizer.joblib'
recommend_recipes_file = 'recommend_recipes.joblib'

joblib.dump(vectorizer, vectorizer_file)
joblib.dump(recommend_recipes, recommend_recipes_file)

print("Model trained and saved successfully.")
