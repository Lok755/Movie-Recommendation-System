# Movie-Recommendation-System
A movie recommendation system built with React suggests movies based on user preferences. Users input their preferences, such as genre or rating, and the system displays recommended movies. React manages the user interface, updating movie suggestions in real-time. Each movie recommendation typically includes details like title, genre, and rating.


import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer

## Load movie data from CSV files
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

## Display the shape (number of rows and columns) of the datasets
print("Shape of credits dataset:", credits.shape)
print("Shape of movies dataset:", movies.shape)

## Display the first few rows and column names of the credits dataset
print(credits.head())
print(credits.columns)

## Display the column names of the movies dataset
print(movies.columns)

## Rename the 'movie_id' column in the credits dataset to 'id' for merging
credits.rename(columns={'movie_id': 'id'}, inplace=True)

## Merge the movies and credits datasets on the 'id' column
movies = movies.merge(credits, on='id')

## Select only the relevant columns for further processing
movies = movies[['id', 'genres', 'title_x', 'keywords', 'overview', 'cast', 'crew']]

## Drop rows with missing values in any of the selected columns
movies.dropna(inplace=True)

## Rename the 'title_x' column to 'title' for clarity
movies.rename(columns={'title_x': 'title'}, inplace=True)

## Define a function to convert stringified list of dictionaries to list of names
def convert(feature):
    name_list = []
    for item in ast.literal_eval(feature):
        name_list.append(item['name'])
    return name_list

## Apply the conversion function to 'genres' and 'keywords' columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

## Define a function to extract top 3 cast members from the list of dictionaries
def convert_cast(feature):
    cast_list = []
    count = 0
    for item in ast.literal_eval(feature):
        if count < 3:
            cast_list.append(item['name'])
            count += 1
        else:
            break
    return cast_list

## Apply the conversion function to the 'cast' column
movies['cast'] = movies['cast'].apply(convert_cast)

## Define a function to extract directors from the crew list of dictionaries
def convert_crew(feature):
    director_list = []
    for item in ast.literal_eval(feature):
        if item['job'] == 'Director':
            director_list.append(item['name'])
    return director_list

## Apply the conversion function to the 'crew' column
movies['crew'] = movies['crew'].apply(convert_crew)

## Tokenize the 'overview' text into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

## Concatenate relevant tags (genres, keywords, cast, crew) into a single list
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

## Clean up tags by removing spaces in individual words
movies['tags'] = movies['tags'].apply(lambda x: [word.replace(" ", "") for word in x])

## Create a new DataFrame with 'id', 'title', and 'tags' columns
new_df = movies[['id', 'title', 'tags']]

## Join the tags into a single string for each movie
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

## Convert the tags to lowercase for uniformity
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

## Use CountVectorizer to convert text data (tags) into a vector representation
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

## Display the shape of the final vectorized representation
print("Vector shape:", vector.shape)
print("Top 10 feature names:", cv.get_feature_names_out()[:10])

import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

## Initialize NLTK (Natural Language Toolkit) for stemming
nltk.download('punkt')  # Download necessary NLTK data (if not already downloaded)

## Initialize Porter Stemmer for word stemming
ps = PorterStemmer()

## Define a stemming function to reduce words to their base (stem) forms
def stemming(text):
    stemmed_words = []
    for word in text.split():  # Convert string to list of words
        stemmed_words.append(ps.stem(word))  # Apply stemming to each word
    return " ".join(stemmed_words)  # Convert list of stemmed words back to string

## Apply stemming to the 'tags' column of the DataFrame
new_df['tags'] = new_df['tags'].apply(stemming)

## Display the stemmed tags for the first movie
print("Stemmed tags for the first movie:")
print(new_df['tags'][0])

## Compute cosine similarity between the vectorized tag representations
similarity = cosine_similarity(vector)

## Display the shape (dimensions) of the similarity matrix
print("Shape of similarity matrix:", similarity.shape)

## Define a function to recommend similar movies based on cosine similarity
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]  # Get the index of the input movie
    distances = similarity[movie_index]  # Retrieve cosine similarity scores for the input movie
    # Find indices and similarity scores for top 5 most similar movies (excluding the input movie itself)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    # Print titles of recommended movies
    print("Recommended movies for", movie, ":")
    for index, _ in movie_list:
        print(new_df.iloc[index].title)

## Example of using the recommend function to suggest movies similar to 'Harry Potter and the Chamber of Secrets'
recommend('Harry Potter and the Chamber of Secrets')

## Remove last 3 rows from the DataFrame (optional step)
new_df = new_df[:-3]

## Convert DataFrame to a dictionary
movies_dict = new_df.to_dict()

## Save the DataFrame dictionary to a pickle file
pickle.dump(movies_dict, open('movies_dict.pkl', 'wb'))

## Save the cosine similarity matrix to a pickle file
pickle.dump(similarity, open('similarity.pkl', 'wb'))

## Example: Retrieve the index of 'Harry Potter and the Chamber of Secrets' in the DataFrame
hp_index = new_df[new_df['title'] == 'Harry Potter and the Chamber of Secrets'].index
print("Index of 'Harry Potter and the Chamber of Secrets' in DataFrame:", hp_index)

