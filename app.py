import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import io

# Load the pre-trained model
model = tf.keras.models.load_model('cnn_model.keras')

def extract_mfcc(file):
    # Load the audio file
    y, sr = librosa.load(file, duration=3, offset=0.5)
    # Extract MFCC features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Define the Streamlit app
def main():
    st.set_page_config(page_title="Stress Detection from Audio", layout="wide")
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.write("Upload an audio file to get predictions.")

    # Main content
    st.title("🎧 Stress Detection from Audio")
    st.markdown("""
    **Welcome to the Stress Detection App!**
    Upload an audio file to analyze whether the speaker is stressed or not. 
    This app uses a Convolutional Neural Network (CNN) to classify audio recordings into two categories: stressed and not stressed.
    """)

    # Upload audio file
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")  # Play the uploaded audio

        # Display a loading spinner while processing
        with st.spinner('Processing your audio file...'):
            try:
                # Extract MFCC from the uploaded audio file
                mfcc = extract_mfcc(uploaded_file)
                mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
                mfcc = np.expand_dims(mfcc, axis=-1) # Add channel dimension if needed

                # Make prediction
                predictions = model.predict(mfcc)

                # Assuming softmax activation in the output layer
                predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the class index with the highest probability

                # Map index to class name and details
                class_names = ["Not Stressed", "Stressed"]
                class_images = {
                    "Not Stressed": "https://th.bing.com/th/id/R.82b8c37a69d495e01b80c356c93e3bc0?rik=XbKjbAvPlYlp8Q&riu=http%3a%2f%2fwww.engineeringwellness.com%2fwp-content%2fuploads%2f2013%2f09%2fbigstock-No-more-Stress-get-some-relax-21897431.jpg&ehk=yy2KRUZWJXgGrUobmDrbMMpzZAk0CV1nLipSceiUjqM%3d&risl=&pid=ImgRaw&r=0",
                    "Stressed": "https://c0.wallpaperflare.com/preview/489/977/224/stress-burnout-man-person.jpg"
                }
                class_descriptions = {
                    "Not Stressed": "The audio suggests the speaker is in a calm state.",
                    "Stressed": "The audio suggests the speaker is experiencing stress."
                }
                class_suggestions = {
                    "Not Stressed": "Keep up the good work! Regular relaxation and mindfulness can help maintain this state.",
                    "Stressed": "Consider practicing stress-relief techniques such as deep breathing exercises, meditation, or talking to a counselor."
                }

                predicted_class = class_names[predicted_class_index]
                
                st.image(class_images[predicted_class],width=500)
                st.write(f"**Prediction:** {predicted_class}")
                st.write(f"**Description:** {class_descriptions[predicted_class]}")
                st.write(f"**Suggestions:** {class_suggestions[predicted_class]}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
