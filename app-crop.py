import streamlit as st
import boto3
import tempfile
from PIL import Image
import pandas as pd
import numpy as np
from aws_extract import extract_text_with_textract
from streamlit_drawable_canvas import st_canvas

def crop_image(image, start_x, start_y, width, height):
    """Crop the image using the provided coordinates and dimensions."""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure coordinates are within bounds and convert to integers
    img_height, img_width = image.shape[:2]
    start_x = max(0, min(int(start_x), img_width))
    start_y = max(0, min(int(start_y), img_height))
    end_x = max(0, min(int(start_x + width), img_width))
    end_y = max(0, min(int(start_y + height), img_height))
    
    # Crop image
    cropped = image[start_y:end_y, start_x:end_x]
    return Image.fromarray(cropped)

def main():
    st.title("Text Extraction Demo V2")
    
    # Add some custom CSS for layout
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
        # Create containers
        image_container = st.container()
        button_container = st.container()
        result_container = st.container()
        
        # Load the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        with image_container:
            # Always show the image preview
            st.image(image, caption='Uploaded Image', width=250)
            
            # Option to select specific area
            use_specific_area = st.checkbox("Select specific area for text extraction")
            
            selected_region = None
            if use_specific_area:
                # Create a canvas for drawing the selection
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Transparent orange fill
                    stroke_width=2,
                    stroke_color="#ff0000",  # Red outline
                    background_image=image,
                    drawing_mode="rect",
                    key="canvas",
                    width=250,  # Match the preview image width
                    height=int(250 * image.height / image.width),  # Maintain aspect ratio
                )
                
                # Get the selected region from the canvas
                if canvas_result.json_data is not None and len(canvas_result.json_data.get("objects", [])) > 0:
                    # Scale the coordinates back to original image size
                    scale_factor = image.width / 250
                    rect = canvas_result.json_data["objects"][-1]  # Get the last drawn rectangle
                    selected_region = {
                        'start_x': rect["left"] * scale_factor,
                        'start_y': rect["top"] * scale_factor,
                        'width': rect["width"] * scale_factor,
                        'height': rect["height"] * scale_factor
                    }
        
        with button_container:
            extract_button = st.button('Extract Text')
        
        with result_container:
            if extract_button:
                with st.spinner('Extracting text...'):
                    # Process either the selected region or the entire image
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        if use_specific_area and selected_region:
                            cropped = crop_image(image,
                                              selected_region['start_x'],
                                              selected_region['start_y'],
                                              selected_region['width'],
                                              selected_region['height'])
                            cropped.save(tmp_file, format='JPEG')
                        else:
                            # Use the entire image
                            image.save(tmp_file, format='JPEG')
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