import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

obj = PorterStemmer()
# Load stopwords once
english_stopwords = stopwords.words("english")

def Transform(text):
    text = text.lower()
    text = word_tokenize(text)

    list = [obj.stem(i) for i in text if i.isalnum() and i not in english_stopwords and i not in string.punctuation]
    return " ".join(list)

# Corrected the loading of model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))  # Corrected variable name

st.title("Spam SMS Detection")
input_text = st.text_area("Enter the SMS, let's find out if it is SPAM")

if st.button("Predict"):

    # preprocess
    Transformed_Messages = Transform(input_text)

    # Vectorize
    vec = tfidf.transform([Transformed_Messages])

    # Predict
    result = model.predict(vec)[0]

    # Display
    if result == 1:
        st.header("IT IS A SPAM MESSAGE")
    else:
        st.header("IT IS NOT A SPAM MESSAGE")
st.header("IT IS A SPAM MESSAGE")
