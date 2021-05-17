import streamlit as st
import pickle
import numpy as np
from sklearn import *

pickle_in = open('model.pkl', 'rb') 
classifier = pickle.load(pickle_in)

def prediction(mood,motivation,attention,irritability,anxiety,sleep_quality,caffeine,active_time):   
    arr = np.array([mood,motivation,attention,irritability,anxiety,sleep_quality,caffeine,active_time]).reshape(1,-1)
    prediction = classifier.predict_proba(arr)
    category = np.argmax(prediction) +1
    if category == 0:
        return "The patient could be tending towards a Mania episode"
    elif category == 1:
        return "The patient could be tending towards a Depression episode"
    else:
        return "The patient is in a normal state"
    # output='{0:.{1}f}'.format(prediction[0][1], 2)

    # if output>str(0.5):
    #     return 'Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output)
    # else:
    #     return 'Your Forest is safe.\n Probability of fire occuring is {}'.format(output)

def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Bipolar Diagnosis Prediction</h1> 
    </div> """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    mood = st.number_input("Enter mood value")

    motivation = st.number_input("Enter motivation value")
    attention = st.number_input("Enter attention value")
    irritability = st.number_input("Enter irritability value")
    anxiety = st.number_input("Enter anxiety value")
    sleep_quality = st.number_input("Enter sleep_quality value")
    caffeine = st.number_input("Enter caffeine value")
    active_time = st.number_input("Enter active_time value")

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(mood,motivation,attention,irritability,anxiety,sleep_quality,caffeine,active_time)
        st.success(result)


if __name__ == '__main__':
    main()
