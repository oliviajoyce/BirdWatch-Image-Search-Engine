#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[105]:


import math
from flask import Flask, render_template, request, redirect, url_for
import json
import os
import re
import nltk

try:
    # Attempt to download NLTK data if not already available
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # NLTK data not found, attempt to download it
    print("NLTK data not found. Attempting to download...")
    nltk.download('punkt')

# Now import the required NLTK modules
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenizes
from nltk.tokenize import word_tokenize
import string
import cv2
from collections import defaultdict

from google.cloud import storage
from google.oauth2 import service_account

import numpy as np
import requests

from datetime import timedelta
import gunicorn


# # Pre-processing

# In[76]:


# Initialize Porter Stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Tokenization and Lowercasing
    tokens = word_tokenize(text.lower())
    # Remove punctuation, stopwords, and perform stemming
    processed_tokens = []
    for token in tokens:
        # Remove punctuation and check if token is not empty after stripping
        token = token.strip(string.punctuation)
        if token != '' and len(token) >= 2:
            # Perform stemming and filter out stopwords
            stemmed_token = stemmer.stem(token)
            if stemmed_token not in stop_words:
                processed_tokens.append(stemmed_token)
    return processed_tokens  # return as list


# # Inverted Index

# In[77]:


def get_static_path(file_name):
    # Assuming notebook is in the same directory as the static folder
    notebook_dir = os.getcwd()
    static_folder = os.path.join(notebook_dir, 'static')
    return os.path.join(static_folder, file_name)


# In[78]:


def load_inverted_index(file_path):
    inverted_index = {}
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            # Split the line into term and postings
            term, postings_str = line.strip().split(":", 1)
            # Convert postings string to list of dictionaries
            postings = eval(postings_str)
            # Create a dictionary entry for the term
            inverted_index[term] = postings
    return inverted_index

# Assuming your inverted index file is located in a 'static' folder in the same directory as your script
# Construct the file path dynamically using the get_static_path function
inverted_index_path = get_static_path('updated_inverted_index.txt')
inverted_index = load_inverted_index(inverted_index_path)


# # Ranking: BM-25

# In[95]:


def idf(term, N, doc_freq):
    return math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

def compute_bm25(inverted_index, query_terms, k1=1.5, b=0.75):
    bm25_scores = []
    N = len(inverted_index)  # Total number of images
    total_text_length = sum(sum(posting['term_frequency'] for posting in postings) for postings in inverted_index.values())
    avgdl = total_text_length / N  # Average document length
    
    for term in query_terms:
        idf_val = idf(term, N, len(inverted_index.get(term, [])))
        if idf_val == 0:
            continue  # Skip terms with IDF of 0
        for posting in inverted_index.get(term, []):
            doc_id = posting['image']
            doc_len = sum(posting['term_frequency'] for posting in inverted_index[term])  # Assuming all terms contribute to the document length
            # Calculate BM25 term score
            term_score = idf_val * (posting['term_frequency'] * (k1 + 1)) / (posting['term_frequency'] + k1 * (1 - b + b * (doc_len / avgdl)))
            bm25_scores.append((term, doc_id, term_score))
    
    return bm25_scores

def rank_bm25(query, inverted_index, k1=1.5, b=0.75):
    query_terms = preprocess(query)  # Assuming the query is already preprocessed
    bm25_scores = compute_bm25(inverted_index, query_terms, k1, b)
    # Filter images that don't contain all query terms
    relevant_docs = set(posting[1] for posting in bm25_scores)
    for term in query_terms:
        if term in inverted_index:
            relevant_docs.intersection_update(posting['image'] for posting in inverted_index[term])
    ranked_docs = [posting for posting in bm25_scores if posting[1] in relevant_docs]
    return ranked_docs


# # Google Cloud Storage

# In[96]:


# Initialize Google Cloud Storage client
notebook_dir = os.getcwd()
key_file_path = os.path.join(notebook_dir, "poised-vial-419810-e53ec981e979.json")
credentials = service_account.Credentials.from_service_account_file(key_file_path)
storage_client = storage.Client(credentials=credentials)
bucket_name = "bird_images_bucket"
bucket = storage_client.bucket(bucket_name)


# In[97]:


def get_image_url(image_id, bucket):
    # Assuming the bucket variable is a Google Cloud Storage Bucket object
    # Check for possible file extensions
    possible_extensions = ['.jpg', '.JPG', '']
    blob = None

    # Try to find a blob that exists with the given extensions
    for ext in possible_extensions:
        blob_path = f"images/{image_id}{ext}"
        blob = bucket.blob(blob_path)
        if blob.exists():
            break
        blob = None  # Reset if not found

    # If no valid blob is found, return None or raise an exception
    if not blob:
        return None  # or raise Exception("Image not found.")

    # Generate a signed URL for the found blob
    url = blob.generate_signed_url(expiration=timedelta(minutes=30))  # URL expires in 30 minutes
    return url


# # Remove Duplicates - OpenCV

# In[98]:


#  Compare images using OpenCV feature extraction
def compute_image_features(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Use a feature extraction technique like ORB
                orb = cv2.ORB_create()
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                return descriptors
            else:
                print(f"Error: Failed to decode image from URL {image_url}")
                return None
        else:
            print(f"Error: Unable to fetch image from URL {image_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

#  Group similar images together
def find_similar_images(inverted_index):
    similar_images = defaultdict(list)
    for keyword, image_info_list in inverted_index.items():
        for image_info in image_info_list:
            image_path = image_info['positions']
            features = compute_image_features(image_path)
            if features is not None:
                similar_images[keyword].append({'image': image_info['image'], 'features': features})
    return similar_images

#Identify representative images from each group
def identify_representative_images(similar_images):
    representative_images = {}
    for keyword, images_info in similar_images.items():
        representative_images[keyword] = []
        # Choose a representative image based on a criteria, e.g., most features
        representative_image = max(images_info, key=lambda x: len(x['features']))
        representative_images[keyword].append(representative_image)
    return representative_images


# # Retrieval: Return Images by User Query

# In[99]:


def filter_images_by_query(query_tokens, selected_country, inverted_index, file_path, run_name):  
    # Define the path to the images metadata file
    metadata_file = get_static_path("textual_surrogate2.txt")
    # Initialize a dictionary to store image metadata
    image_metadata = {}
    # Read the metadata file and parse its contents as JSON
    with open(metadata_file, "r", encoding='utf-8') as file:
        data = json.load(file)
        # Calculate BM25 scores for images
        bm25_scores = rank_bm25(' '.join(query_tokens), inverted_index)
        write_results(bm25_scores, file_path, run_name)
        # Get the image metadata based on the ranked image IDs
        for term, image_id, score in bm25_scores:
            image_entry = next((entry for entry in data if entry['id'] == image_id), None)
            if image_entry:
                image_metadata[image_id] = image_entry
                image_metadata[image_id]['image_url'] = get_image_url(image_id,bucket)  # Update image URL
                image_metadata[image_id]['original_caption'] = image_entry['original_caption']
                image_metadata[image_id]['original_country'] = image_entry['original_country']
              
                
    # Step 6: Filter out duplicate images based on image features
    def filter_duplicates(image_metadata):
        unique_image_metadata = []
        seen_features = set()
        for image in image_metadata.values():
            features = compute_image_features(image['image_url'])  #image_path
            if features is not None:
                features_tuple = tuple(tuple(row) for row in features)  # Convert NumPy array to tuple of tuples
                hash_value = hash(features_tuple)
                if hash_value not in seen_features:
                    unique_image_metadata.append(image)
                    seen_features.add(hash_value)
        return unique_image_metadata

    # Filter out duplicate images
    unique_image_metadata = filter_duplicates(image_metadata)

    return unique_image_metadata


# # Output Model Results per Query

# In[100]:


def write_results(results, file_path, run_name):
    print("Results:", results)  # Add this line to inspect the results
    with open(file_path, 'w') as f:
        rank = 1  # Initialize rank counter
        for result in results:
            if result[2] != 0.0:  # Exclude results with score 0.0
                query_id =  str(result[0])
                document_id =  str(result[1])
                score = result[2]
                # Write in TREC format: <query_id> <Q0> <doc_id> <rank> <score> <run_id>
                f.write(f"{query_id} Q0 {document_id} {rank} {score} {run_name}\n")
                rank += 1  # Increment rank for the next document



# # Web Application - Flask

# In[101]:


# Parse the textual_surrogate2.txt file to extract unique country values
textual_surrogate_file= get_static_path("original_textual_surrogate.txt")
with open(textual_surrogate_file, 'r') as f:
    data = json.load(f)

# Extract unique country values
unique_countries = sorted(set(entry['country'] for entry in data))

# Modify the image entries to include original captions
for entry in data:
    entry['original_caption'] = entry['caption']


# In[102]:


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('combined.html', countries = unique_countries)

@app.route('/search-results', methods=['POST'])
def handle_form():
    if request.method == 'POST':
        # Get the search keywords and selected country from the form
        search_keywords = request.form.get('message')
        selected_country = request.form.get('country')

        # Combine search keywords and selected country if both are provided
        if search_keywords and selected_country:
            search_query = f"{search_keywords} {selected_country}"
        else:
            # Use either search keywords or selected country if one of them is provided
            search_query = search_keywords or selected_country

        # Preprocess the combined search query
        processed_query = preprocess(search_query)

        # Filter images based on the processed query
        file_path = get_static_path('results.txt')
        image_metadata = filter_images_by_query(processed_query, selected_country, inverted_index, file_path, run_name=search_query)
        num_images = len(image_metadata)

        # Render the gallery page with the filtered images
        return render_template('combined.html', country=search_query, num_images=num_images, images=image_metadata, countries=unique_countries, show_dynamic=True)
    else:
        # If the method is not POST, render the gallery page without the dynamic content
        return render_template('combined.html', show_dynamic=False, countries=unique_countries)


#app.run(debug=True, port=8080, use_reloader=False)


# In[ ]:




