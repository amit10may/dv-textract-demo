import streamlit as st
import boto3
import tempfile
from PIL import Image
import pandas as pd
import numpy as np
from aws_extract import extract_text_with_textract
from streamlit_drawable_canvas import st_canvas
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from ultralytics import YOLO

def crop_image(image, start_x, start_y, width, height):
    """Crop the image using the provided coordinates and dimensions."""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # Ensure coordinates are within bounds and convert to integers
    img_height, img_width = image_array.shape[:2]
    start_x = max(0, min(int(start_x), img_width))
    start_y = max(0, min(int(start_y), img_height))
    end_x = max(0, min(int(start_x + width), img_width))
    end_y = max(0, min(int(start_y + height), img_height))
    
    # Crop image
    cropped = image_array[start_y:end_y, start_x:end_x]
    
    # Convert back to PIL Image
    return Image.fromarray(cropped)

def extract_text_with_easyocr(image_path):
    """Extract text using EasyOCR."""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    
    extracted_data = [
        {
            'text': text,
            'confidence': confidence * 100
        }
        for (bbox, text, confidence) in results
    ]
    return extracted_data

def extract_text_with_paddleocr(image_path):
    """Extract text using PaddleOCR."""
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    results = ocr.ocr(image_path)
    
    extracted_data = [
        {
            'text': text,
            'confidence': confidence * 100
        }
        for (bbox, (text, confidence)) in results[0] if results[0]
    ]
    return extracted_data

def extract_text_with_trocr(image_path, processor=None, model=None):
    """Extract text using TrOCR."""
    # If processor and model aren't provided, use default ones
    if processor is None or model is None:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return [{'text': generated_text, 'confidence': 95}]

def extract_text_with_yolo_ocr(image_path):
    """Extract text using YOLO + EasyOCR."""
    # Load YOLO model
    model = YOLO('yolov8x.pt')
    
    # Load EasyOCR
    reader = easyocr.Reader(['en'])
    
    # Detect text regions with YOLO
    results = model(image_path)
    
    extracted_data = []
    image = Image.open(image_path)
    
    # Process each detected region
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        
        # Crop region
        region = image.crop((x1, y1, x2, y2))
        
        # Perform OCR on region
        ocr_result = reader.readtext(np.array(region))
        
        for _, text, confidence in ocr_result:
            extracted_data.append({
                'text': text,
                'confidence': confidence * 100
            })
    
    return extracted_data

def extract_text_with_yolo_trocr(image_path):
    """Extract text using YOLO + TrOCR instead of EasyOCR"""
    try:
        # Load models
        yolo_model = YOLO('yolov8x.pt')
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
        
        # Detect regions with YOLO
        results = yolo_model(image_path)
        
        extracted_data = []
        image = Image.open(image_path)
        
        # Process each detected region
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            
            # Crop region
            region = image.crop((int(x1), int(y1), int(x2), int(y2)))
            
            # OCR on region
            pixel_values = processor(region, return_tensors="pt").pixel_values
            generated_ids = ocr_model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if text.strip():  # Only add non-empty text
                extracted_data.append({
                    'text': text,
                    'confidence': 95  # Estimated confidence
                })
        
        return extracted_data if extracted_data else [{'text': 'No text detected', 'confidence': 0}]
        
    except Exception as e:
        st.error(f"Error in YOLO+TrOCR processing: {str(e)}")
        return [{'text': f'Error: {str(e)}', 'confidence': 0}]

def main():
    st.title("Text Extraction Demo V3")
    
    # Model selection
    model_type = st.selectbox(
        "Select extraction model",
        ["AWS Textract", 
         "TrOCR (Handwritten)", 
         "TrOCR (Printed)",
         "YOLO+TrOCR"]
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Create containers
        image_container = st.container()
        button_container = st.container()
        result_container = st.container()
        
        # Load the image
        image = Image.open(uploaded_file)
        
        with image_container:
            # Always show the image preview
            st.image(image, caption='Uploaded Image', width=250)
            
            # Option to select specific area
            use_specific_area = st.checkbox("Select specific area for text extraction")
            
            selected_region = None
            if use_specific_area:
                # Calculate canvas dimensions
                canvas_width = 250
                canvas_height = int(canvas_width * image.height / image.width)
                
                # Create canvas for drawing the selection
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#ff0000",
                    background_image=image,
                    drawing_mode="rect",
                    key="canvas",
                    width=canvas_width,
                    height=canvas_height,
                )
                
                # Get the selected region from the canvas
                if (canvas_result is not None and 
                    canvas_result.json_data is not None and 
                    "objects" in canvas_result.json_data and 
                    canvas_result.json_data["objects"]):
                    # Scale the coordinates back to original image size
                    scale_factor = image.width / canvas_width
                    rect = canvas_result.json_data["objects"][-1]
                    selected_region = {
                        'start_x': rect["left"] * scale_factor,
                        'start_y': rect["top"] * scale_factor,
                        'width': rect["width"] * scale_factor,
                        'height': rect["height"] * scale_factor
                    }
        
        with button_container:
            if st.button('Extract Text'):
                if use_specific_area and selected_region is None:
                    st.warning("Please draw a selection area first.")
                    return
                
                with st.spinner('Extracting text...'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        if use_specific_area and selected_region is not None:
                            cropped = crop_image(image,
                                              selected_region['start_x'],
                                              selected_region['start_y'],
                                              selected_region['width'],
                                              selected_region['height'])
                            cropped.save(tmp_file, format='JPEG')
                        else:
                            image.save(tmp_file, format='JPEG')
                        temp_path = tmp_file.name
                    
                    try:
                        # Extract text using selected model
                        if model_type == "AWS Textract":
                            extracted_data = extract_text_with_textract(temp_path)
                        elif model_type == "TrOCR (Handwritten)":
                            extracted_data = extract_text_with_trocr(temp_path)
                        elif model_type == "TrOCR (Printed)":
                            processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')
                            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-stage1')
                            extracted_data = extract_text_with_trocr(temp_path, processor, model)
                        else:  # YOLO+TrOCR
                            extracted_data = extract_text_with_yolo_trocr(temp_path)
                        
                        # Display results
                        if extracted_data:
                            df = pd.DataFrame(extracted_data)
                            df.columns = ['Text', 'Confidence (%)']
                            df['Confidence (%)'] = df['Confidence (%)'].round(2)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("No text was detected in the image.")
                            
                    except Exception as e:
                        st.error(f"Error during text extraction: {str(e)}")

if __name__ == "__main__":
    main()