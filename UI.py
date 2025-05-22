import streamlit as st
import pickle

# Function to load the selected model
def load_model(model_name):
    model_path = f'{model_name}.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Title of the web app
    st.title('Gender Based Violence')

    # Set custom CSS for the background image
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url('');
        background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Subheader
    st.subheader('Welcome! Select a model and input features for prediction.')

    # Dropdown to select the model
    model_options = ['RandomForest', 'LogisticRegression']
    selected_model = st.selectbox('Select Model', model_options)

    # Load the selected model
    model = load_model(selected_model)

    # User input for features
    st.header('Feature Input')
    tweet = st.text_area('Tweet', 'Write your tweet here...')

    # Button for predictions
    clicked = st.button('Get Predictions')

    # Perform predictions when the button is clicked
    if clicked:
        # Perform predictions using the selected model
        vectorizer_path = f'tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        tweet_vectorized = vectorizer.transform([tweet])  # Convert the tweet to a 2D array
        prediction = model.predict(tweet_vectorized)

        # Display the prediction result
        st.header('Prediction')
        st.write(f'The prediction result is: {prediction[0]}')

if __name__ == '__main__':
    main()
