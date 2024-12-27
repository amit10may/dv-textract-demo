import streamlit as st
from PIL import Image
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sklearn.metrics.pairwise import cosine_similarity

def get_image_similarity(img1, img2):
    """Calculate visual similarity between two images using average hash"""
    # Convert PIL images to cv2 format and resize
    img1_array = cv2.resize(np.array(img1), (64, 64))
    img2_array = cv2.resize(np.array(img2), (64, 64))
    
    # Convert to BGR
    img1_array = cv2.cvtColor(img1_array, cv2.COLOR_RGB2BGR)
    img2_array = cv2.cvtColor(img2_array, cv2.COLOR_RGB2BGR)
    
    # Calculate image hashes
    hasher = cv2.img_hash.AverageHash_create()
    hash1 = hasher.compute(img1_array)
    hash2 = hasher.compute(img2_array)
    
    # Calculate similarity
    similarity = 1 - (cv2.norm(hash1, hash2) / len(hash1))
    return float(similarity)

def extract_text(image, processor, model):
    """Extract text from image using TrOCR"""
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def main():
    st.title("Image Comparison Tool")
    
    # Create two columns for side-by-side image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Image")
        image1 = st.file_uploader("Upload first image", type=['png', 'jpg', 'jpeg'], key="img1")
        
    with col2:
        st.subheader("Second Image")
        image2 = st.file_uploader("Upload second image", type=['png', 'jpg', 'jpeg'], key="img2")
    
    if image1 and image2:
        # Load and display images
        img1 = Image.open(image1)
        img2 = Image.open(image2)
        
        col1.image(img1, use_column_width=True)
        col2.image(img2, use_column_width=True)
        
        if st.button("Compare Images"):
            with st.spinner("Analyzing images..."):
                # Visual similarity
                visual_similarity = get_image_similarity(img1, img2)
                
                # Text similarity
                try:
                    # Load OCR models
                    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
                    ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
                    text_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Extract text
                    text1 = extract_text(img1, processor, ocr_model)
                    text2 = extract_text(img2, processor, ocr_model)
                    
                    # Get text embeddings and similarity
                    emb1 = text_model.encode([text1])[0]
                    emb2 = text_model.encode([text2])[0]
                    text_similarity = cosine_similarity([emb1], [emb2])[0][0]
                    
                    # Combined similarity
                    combined_similarity = (0.6 * visual_similarity + 0.4 * text_similarity)
                    
                    # Display results
                    st.subheader("Comparison Results:")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Visual Similarity", f"{visual_similarity:.2%}")
                    with col2:
                        st.metric("Text Similarity", f"{text_similarity:.2%}")
                    with col3:
                        st.metric("Combined Similarity", f"{combined_similarity:.2%}")
                    
                    # Display extracted text
                    st.subheader("Extracted Text:")
                    text_col1, text_col2 = st.columns(2)
                    with text_col1:
                        st.text_area("Text from Image 1:", text1, height=100)
                    with text_col2:
                        st.text_area("Text from Image 2:", text2, height=100)
                    
                except Exception as e:
                    st.error(f"Error in text comparison: {str(e)}")
                    st.metric("Visual Similarity Only", f"{visual_similarity:.2%}")

if __name__ == "__main__":
    main()
