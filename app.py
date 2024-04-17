# imports
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from tabulate import tabulate
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

app = Flask(__name__)

# Download NLTK resources
# nltk.download('stopwords')

# Load and preprocess
# Load and preprocess data
data = pd.read_csv("DataAfterPreAndFeatures.csv")
data['Tokens'] = data['Tokens'].fillna('').astype(str)


# Build the Tf-Idf model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Tokens'])

# User location
user_location = "something"
geolocator = Nominatim(user_agent="place_recommendation")
location = geolocator.geocode(user_location)
user_latitude = location.latitude
user_longitude = location.longitude

# Preprocess user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^A-Za-z]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    user_description = "another thing"
    user_tokens = preprocess_text(user_description)
    preprocessed_user_input = vectorizer.transform([user_tokens])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix, preprocessed_user_input)

    # Add similarity scores to the DataFrame
    data['Similarity'] = similarities.flatten()

    # Calculate distances from user location in kilometers
    data['Distance'] = data.apply(lambda row: geodesic((user_latitude, user_longitude), (row['lat'], row['lng'])).kilometers, axis=1)

    # Filter places within 5 kilometers
    filtered_data = data[data['Distance'] <= 5]

    # Sort by similarity, distance, and rating
    sorted_data = filtered_data.sort_values(by=['Similarity', 'Rating'], ascending=[False, False])

    # Filter top 3 recommendations
    recommendations = sorted_data.head(3)

    # Format the recommendations as a table
    table_columns = ['Name',  'Rating', 'Google Maps URL', 'Image']
    table_data = recommendations[table_columns]

    # Render the HTML template with recommendations table
    return render_template('page1.html', table=tabulate(table_data, headers='keys', tablefmt='html'))

@app.route('/where', methods=['POST', 'GET'])
def whereto():
    location = request.form.get('location')
    to = request.form.get('to')
    return render_template('page2.html')

if __name__ == '__main__':
    app.run(debug=True)
    
