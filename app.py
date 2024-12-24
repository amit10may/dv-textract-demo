import streamlit as st
import boto3
import tempfile
from PIL import Image
import pandas as pd
from aws_extract import extract_text_with_textract

def main():
    st.title("Text Extraction Demo V1")
    
    # Add some custom CSS to help with layout
    st.markdown("""
        <style>
        .main > div {
            padding: 1em;
        }
        .stButton > button {
            width: 100%;
            margin-top: 1em;
            margin-bottom: 1em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Create three separate containers
        image_container = st.container()
        button_container = st.container()
        result_container = st.container()
        
        with image_container:
            # Display the uploaded image with reduced width
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=250)
        
        with button_container:
            # Add extract button
            extract_button = st.button('Extract Text')
        
        with result_container:
            if extract_button:
                with st.spinner('Extracting text...'):
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    try:
                        # Extract text using your existing function
                        extracted_data = extract_text_with_textract(temp_path)
                        
                        # Convert to DataFrame for nice display
                        df = pd.DataFrame(extracted_data)
                        df.columns = ['Text', 'Confidence (%)']
                        df['Confidence (%)'] = df['Confidence (%)'].round(2)
                        
                        # Display results
                        st.subheader("Extracted Text")
                        st.dataframe(df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during text extraction: {str(e)}")

if __name__ == "__main__":
    main()