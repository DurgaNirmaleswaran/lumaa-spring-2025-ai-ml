#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re


# In[14]:


# Text preprocessing function

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return text


# In[18]:


# Extracts names from list fields (for genres and keywords)

def extract_list_fields(field_data, key='name'):
    try:
        data = ast.literal_eval(str(field_data))
        return ' '.join([item[key] for item in data if key in item])
    except:
        return ''


# In[22]:


# Prepares the dataframe by processing text fields and computing TF-IDF features

def prepare_data(df):
    df['processed_overview'] = df['overview'].apply(preprocess_text)
    df['processed_genres'] = df['genres'].apply(lambda x: extract_list_fields(x))
    df['processed_keywords'] = df['keywords'].apply(lambda x: extract_list_fields(x))
    df['combined_features'] = (
        df['processed_overview'] + ' ' +
        df['processed_genres'] + ' ' +
        df['processed_keywords']
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return df, tfidf, tfidf_matrix


# In[27]:


# Predicts movie recommendations based on the user input

def get_recommendations(user_input, df, tfidf, tfidf_matrix, n_recommendations=5):
    processed_input = preprocess_text(user_input)
    user_vector = tfidf.transform([processed_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    n_recommendations = min(n_recommendations, len(df))
    top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
    
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'title': df.iloc[idx]['original_title'],
            'overview': df.iloc[idx]['overview'],
            'similarity_score': similarity_scores[idx],
            'genres': df.iloc[idx]['genres']
        })
    return recommendations


# In[33]:


# Loads and preprocesses movie data
def load_data(filepath, num_rows=500):
    try:
        df = pd.read_csv(filepath, nrows=num_rows)
        columns_needed = ['original_title', 'overview', 'genres', 'keywords']
        df = df[columns_needed].copy()
        df = df.dropna(subset=['overview'])
        df = df.sort_values('original_title').reset_index(drop=True)
        print(f"Successfully loaded {len(df)} movies.")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


# In[42]:


def load_initial_movies(file_path, num_rows=300):
   
    """
    Load initial set of movies from CSV file that is taking a small set of data for faster processing
    """
    
    try:
        df = pd.read_csv(file_path, nrows=num_rows)
        print(f"\nSuccessfully loaded {len(df)} movies")
        return df
    except FileNotFoundError:
        print(f"\nError: Could not find the file {file_path}")
        return None
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        return None


# In[45]:


def prepare_movie_data(df):
   
    """
    Prepare movie data for recommendations
    """
    
    # Creates TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Creates TF-IDF matrix from movie overviews
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    
    return tfidf, tfidf_matrix


# In[51]:


def get_recommendations(query, df, tfidf, tfidf_matrix):
    
    """
    Get movie recommendations based on user query
    """
    
    # Transforms user query
    query_vec = tfidf.transform([query])
    
    # Calculates similarity scores
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Obtaining indices of top 5 movies
    top_indices = similarity_scores.argsort()[-5:][::-1]
    
    # Creating recommendations list
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'title': df.iloc[idx]['title'],
            'overview': df.iloc[idx]['overview'],
            'genres': df.iloc[idx]['genres'],
            'similarity_score': similarity_scores[idx]
        })
    
    return recommendations


# In[56]:


def print_detailed_recommendations(recommendations, user_query):
   
    """
    Print detailed movie recommendations
    """
    
    print("\n=== DETAILED MOVIE RECOMMENDATIONS ===")
    print(f"\nBased on your preference: '{user_query}'\n")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title'].upper()}")
        print(f"\nSimilarity Score: {rec['similarity_score']:.3f}")

            
        print("\nOverview:")
        print(rec['overview'])
        print("\n" + "=" * 80)


# In[60]:


def get_user_preferences():
   
    """
    Get user movie preferences
    """
    
    print("\n=== MOVIE RECOMMENDATION SYSTEM ===")
    print("\nDescribe the kind of movie you're looking for!")
    print("Examples:")
    print("- 'I love action movies with lots of explosions and car chases'")
    print("- 'Looking for romantic comedies set in New York'")
    print("- 'Sci-fi movies about time travel'")
    print("- 'Dark psychological thrillers with plot twists'\n")
    return input("Your movie preference: ").strip()


# In[61]:


def main():
   
    # First load the initial set of movies
    
    print("Loading initial movie database...")
    df = load_initial_movies("tmdb_5000_movies.csv", num_rows=500)
    
    if df is not None:
        
        # Prepare the data
        print("Preparing recommendation system...")
        tfidf, tfidf_matrix = prepare_movie_data(df)
        
        while True:
            user_query = get_user_preferences()
            if not user_query:
                print("\nPlease enter a description of your movie preferences.")
                continue
            
            recommendations = get_recommendations(user_query, df, tfidf, tfidf_matrix)
            print_detailed_recommendations(recommendations, user_query)
            
            try_again = input("\nWould you like to try another search? (yes/no): ").lower()
            if try_again not in ['yes', 'y']:
                print("\nThank you for using the Movie Recommendation System!")
                break

if __name__ == "__main__":
    main()


# In[ ]:




