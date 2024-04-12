# Movie-Recommendation-System
A movie recommendation system built with React suggests movies based on user preferences. Users input their preferences, such as genre or rating, and the system displays recommended movies. React manages the user interface, updating movie suggestions in real-time. Each movie recommendation typically includes details like title, genre, and rating.


import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer

## Load movie data from CSV files
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Display the shape (number of rows and columns) of the datasets
print("Shape of credits dataset:", credits.shape)
print("Shape of movies dataset:", movies.shape)

# Display the first few rows and column names of the credits dataset
print(credits.head())
print(credits.columns)

# Display the column names of the movies dataset
print(movies.columns)

# Rename the 'movie_id' column in the credits dataset to 'id' for merging
credits.rename(columns={'movie_id': 'id'}, inplace=True)

# Merge the movies and credits datasets on the 'id' column
movies = movies.merge(credits, on='id')

# Select only the relevant columns for further processing
movies = movies[['id', 'genres', 'title_x', 'keywords', 'overview', 'cast', 'crew']]

# Drop rows with missing values in any of the selected columns
movies.dropna(inplace=True)

# Rename the 'title_x' column to 'title' for clarity
movies.rename(columns={'title_x': 'title'}, inplace=True)

# Define a function to convert stringified list of dictionaries to list of names
def convert(feature):
    name_list = []
    for item in ast.literal_eval(feature):
        name_list.append(item['name'])
    return name_list

# Apply the conversion function to 'genres' and 'keywords' columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Define a function to extract top 3 cast members from the list of dictionaries
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

# Apply the conversion function to the 'cast' column
movies['cast'] = movies['cast'].apply(convert_cast)

# Define a function to extract directors from the crew list of dictionaries
def convert_crew(feature):
    director_list = []
    for item in ast.literal_eval(feature):
        if item['job'] == 'Director':
            director_list.append(item['name'])
    return director_list

# Apply the conversion function to the 'crew' column
movies['crew'] = movies['crew'].apply(convert_crew)

# Tokenize the 'overview' text into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Concatenate relevant tags (genres, keywords, cast, crew) into a single list
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Clean up tags by removing spaces in individual words
movies['tags'] = movies['tags'].apply(lambda x: [word.replace(" ", "") for word in x])

# Create a new DataFrame with 'id', 'title', and 'tags' columns
new_df = movies[['id', 'title', 'tags']]

# Join the tags into a single string for each movie
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert the tags to lowercase for uniformity
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Use CountVectorizer to convert text data (tags) into a vector representation
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Display the shape of the final vectorized representation
print("Vector shape:", vector.shape)
print("Top 10 feature names:", cv.get_feature_names_out()[:10])
