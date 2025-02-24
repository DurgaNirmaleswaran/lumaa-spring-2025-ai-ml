# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

---

## Overview

- This project is a simple yet functional content-based recommendation system designed for an AI/ML internship challenge. The system allows users to input a short text description of their preferences—for example, "I love thrilling action movies set in space, with a comedic twist" and returns a ranked list of similar items from a curated movie dataset. 

- The dataset includes various fields such as budget, genres, keywords, original title, overview, and production companies. Using a straightforward approach based on TF-IDF vectorization and cosine similarity, the system converts both the user query and each movie's textual information into numerical vectors. 

- It then computes the similarity between the query and every movie in the dataset, returning the top recommendations based on the highest similarity scores. The project is implemented in Python and organized into clear, modular sections for data loading, preprocessing, vectorization, similarity computation, and result presentation. 

- This prototype is meant to be lightweight, efficient, and easy to understand, perfect for demonstrating core machine learning and recommendation concepts within a limited timeframe. Additionally, the project includes a README with setup instructions, a brief video demo of the system in action, and details on the expected monthly salary as part of the challenge submission requirements.


## Dataset
- **Source**: The dataset used is based on movie metadata and includes fields such as:
  - `budget`
  - `genres`
  - `homepage`
  - `id`
  - `keywords`
  - `original_language`
  - `original_title`
  - `overview`
  - `popularity`
  - `production_companies`
- **Preparation**: You can use a publicly available dataset (e.g., [TMDB 5000 Movies](https://www.kaggle.com/tmdb/tmdb-movie-metadata)) and include it in your forked repository. For faster prototyping, the code loads only a subset (e.g., 300–500 rows).

- tmdb_5000_movies.csv -  CSV file containing movie data
- main.py - Main Python script with recommendation system code
- README.md - This README file
- demo.md - link to a video demo

## Requirements
- **Python Version**: 3.6+
- **Dependencies**:
  - pandas
  - numpy
  - scikit-learn
  - (Standard libraries: ast, re)

Install the necessary libraries using:
```bash
pip install pandas numpy scikit-learn
```

## How It Works
**Data Loading & Preprocessing:**

The system loads a CSV file containing movie data. It preprocesses text fields such as overview, genres, and keywords by cleaning the text (removing punctuation, converting to lowercase) and extracting useful information from list-like fields. It creates a combined text feature from the processed fields.

**Feature Extraction:**

The combined text is vectorized using TF-IDF, creating a numerical representation for each movie.

## Similarity Calculation & Recommendations:

When a user inputs their movie preference, the system transforms the input into the same TF-IDF vector space. Cosine similarity is computed between the user query and every movie vector. The top 5 movies with the highest similarity scores are returned as recommendations.

## Running the Code
- **Prepare the Dataset:** Ensure that tmdb_5000_movies.csv is in the project directory.
- **Run the Script:** python main.py
- **Follow the Prompts:** The program will prompt you to enter your movie preference, then display detailed recommendations including the movie title, overview, genres, and similarity score.

## Example Usage

**=== MOVIE RECOMMENDATION SYSTEM ===**

Describe the kind of movie you're looking for!
Examples:
- "I love action movies with lots of explosions and car chases"
- "Looking for romantic comedies set in New York"
- "Sci-fi movies about time travel"
- "Dark psychological thrillers with plot twists"

Your movie preference: I love thrilling action movies set in space, with a comedic twist

**=== DETAILED MOVIE RECOMMENDATIONS ===**

Based on your preference: 'I love thrilling action movies set in space, with a comedic twist'

1. GUARDIANS OF THE GALAXY
   Similarity Score: 0.245
   Overview: A group of intergalactic criminals are forced to work together to stop a fanatical warrior...
   Genres: [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}]
...

## Salary Expectation
20-40$/hour 

