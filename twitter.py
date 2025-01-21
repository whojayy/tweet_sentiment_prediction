import numpy as np
import pandas as pd
import re     # re stands for regular expression
from nltk.corpus import stopwords   # nltk stands for natural language tool kit
from nltk.stem.porter import PorterStemmer    # Used for steming
from sklearn.feature_extraction.text import TfidfVectorizer   # fir converting textual data to numerical data
from sklearn.model_selection import train_test_split    # Forspliting the dataset
from sklearn.linear_model import LogisticRegression   # Importing ML model
from sklearn.metrics import accuracy_score    # To calulate the performance of our model
import pickle


# Function for stemming and preprocessing of user input
def stemming_user_data(content):
    """
    This function takes a string input, removes special characters, converts to lowercase,
    removes stopwords, and applies stemming.
    """
    port_stem = PorterStemmer()
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    # Tokenize the sentence into words
    stemmed_content = stemmed_content.split()
    # Remove stopwords and apply stemming
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    # Rejoin the words into a single string
    return ' '.join(stemmed_content)


# Load the trained vectorizer
with open('/Users/jay/Developer/Projects/resume_projects/twitter/vectorizer.sav', 'rb') as file:
    loaded_vectorizer = pickle.load(file)



def predict_user_sentiment(user_input):
    """
    Predict the sentiment of a user-provided sentence.
    """
    try:
        # Step 1: Preprocess the input
        preprocessed_input = stemming_user_data(user_input)

        # Step 2: Load the trained vectorizer
        with open('/Users/jay/Developer/Projects/resume_projects/twitter/vectorizer.sav', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)

        # Transform the preprocessed input into a vector
        input_vector = loaded_vectorizer.transform([preprocessed_input])

        # Step 3: Load the trained model
        with open('/Users/jay/Developer/Projects/resume_projects/twitter/trained_model.sav', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Step 4: Predict sentiment
        prediction = loaded_model.predict(input_vector)

        # Step 5: Interpret the result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"

        # Return the result
        return {
            "Input": user_input,
            "Preprocessed Input": preprocessed_input,
            "Predicted Sentiment": sentiment
        }

    except Exception as e:
        # Handle any errors and return the error message
        return {"Error": str(e)}
    



user_sentence = "I  good"
result = predict_user_sentiment(user_sentence)
print(result)


user_sentence = "So tired today, enough of work"
result = predict_user_sentiment(user_sentence)
print(result)


user_sentence = "Yay Trump won"
result = predict_user_sentiment(user_sentence)
print(result)



