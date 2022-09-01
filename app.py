import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string

from nltk.stem.porter import PorterStemmer # stemming 
ps=PorterStemmer()

def transform_text(text):
    
    text=text.lower() #lowercase
    text=nltk.word_tokenize(text) #tokenization
    
    #Removig special character
    temp=[]
    for i in text :
        if i.isalnum():
            temp.append(i)
            
    #copyng list
    text=temp[:]
    temp.clear()
    
    # Removing stop word and punctuation

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)
    
    #copyng list
    text=temp[:]
    temp.clear()
    
    #stemming
    for i in text:
        temp.append(ps.stem(i))
    
    
    return " ".join(temp) 
    


tfidf=pickle.load(open('vectorizer.pkl','rb')) # read binary mode
model=pickle.load(open('model.pkl','rb')) # read binary mode

st.title("Email /SMS Spam Classifier")

# input
input_sms=st.text_input("Enter The Message")

if st.button('Predict'):
    # Work on 4 following process:

    # Preprocess
    transformed_sms=transform_text(input_sms)

    # Vectorize
    vector_input=tfidf.transform([transformed_sms])

    # Predict
    result=model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")