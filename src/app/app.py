import requests
import streamlit as st

API_URL = "http://model_inference_endpoint:8000/get-prediction/"

# API_URL = "http://model_inference_endpoint:8000/get-prediction/"


def get_prediction(input_text):
    response = requests.post(
        API_URL,
        json={"input_text": input_text}
    )

    print("Status code:", response.status_code)
    print("Raw response:", response.text)

    if response.status_code == 200:
        return response.json()
    else:
        return None

st.title("Tweet Classifier")

text = st.text_input("Enter a tweet")

if st.button("Predict"):
    if text == "":
        st.write("Please enter text")
    else:
        result = get_prediction(text)
        st.write("Result:", result)

        if result is not None:
            st.write("Prediction:", result["prediction"])
        else:
            st.write("Something went wrong")