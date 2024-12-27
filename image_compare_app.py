import streamlit as st
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import torch
from torchvision import models as torch_models
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = None

def load_models():
    """Load all required models"""
    if st.session_state.models is None:
        models_dict = {}
        try:
            models_dict['processor'] = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
            models_dict['ocr_model'] = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
            models_dict['text_model'] = SentenceTransformer('all-MiniLM-L6-v2')
            resnet = torch_models.resnet18(pretrained=True)
            models_dict['resnet'] = torch.nn.Sequential(*(list(resnet.children())[:-1]))
            models_dict['resnet'].eval()
            st.session_state.models = models_dict
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None
    return st.session_state.models

def get_image_similarity_ssim(img1, img2):
    """Compare images using Structural Similarity Index"""
    # Convert PIL to cv2 and resize
    img1_array = cv2.resize(np.array(img1), (200, 200))
    img2_array = cv2.resize(np.array(img2), (200, 200))
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate SSIM
    score = structural_similarity(img1_gray, img2_gray)
    return float(score)

def get_image_similarity_orb(img1, img2):
    """Compare images using ORB features matching"""
    # Convert PIL to cv2
    img1_array = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2_array = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1_array, None)
    kp2, des2 = orb.detectAndCompute(img2_array, None)
    
    if des1 is None or des2 is None:
        return 0.0
    
    # Create BF Matcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Calculate similarity score
    similarity = len(matches) / max(len(kp1), len(kp2))
    return float(similarity)

def get_image_similarity_histogram(img1, img2):
    """Compare images using color histogram correlation"""
    # Convert PIL to cv2
    img1_array = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_array = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    # Calculate histograms
    hist1 = cv2.calcHist([img1_array], [0, 1, 2], None, [8, 8, 8], 
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_array], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Calculate correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(correlation)

def extract_text(image, models):
    """Extract text from image using TrOCR"""
    try:
        pixel_values = models['processor'](image, return_tensors="pt").pixel_values
        generated_ids = models['ocr_model'].generate(pixel_values)
        text = models['processor'].batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    except Exception as e:
        st.warning(f"OCR Error: {str(e)}")
        return ""

def get_text_similarity(text1, text2, model):
    """Calculate text similarity using sentence embeddings"""
    if not text1 or not text2:
        return 0.0
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])

def main():
    st.set_page_config(page_title="Image Comparison Tool", layout="wide")
    st.title("Advanced Image Comparison Tool")
    
    # Load models
    with st.spinner("Loading models... This may take a minute..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # Sidebar for method selection
    st.sidebar.title("Comparison Settings")
    comparison_methods = st.sidebar.multiselect(
        "Select Comparison Methods",
        ["Structural Similarity (SSIM)", 
         "Feature Matching (ORB)", 
         "Color Histogram",
         "Text Similarity (OCR)"],
        default=["Structural Similarity (SSIM)", "Feature Matching (ORB)"]
    )
    
    # Weight adjustments in sidebar
    st.sidebar.subheader("Adjust Weights")
    weights = {}
    if "Structural Similarity (SSIM)" in comparison_methods:
        weights['ssim'] = st.sidebar.slider("SSIM Weight", 0.0, 1.0, 0.3, 0.1)
    if "Feature Matching (ORB)" in comparison_methods:
        weights['orb'] = st.sidebar.slider("ORB Weight", 0.0, 1.0, 0.3, 0.1)
    if "Color Histogram" in comparison_methods:
        weights['histogram'] = st.sidebar.slider("Histogram Weight", 0.0, 1.0, 0.2, 0.1)
    if "Text Similarity (OCR)" in comparison_methods:
        weights['text'] = st.sidebar.slider("Text Weight", 0.0, 1.0, 0.2, 0.1)
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
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
                scores = {}
                
                # Calculate selected similarity scores
                if "Structural Similarity (SSIM)" in comparison_methods:
                    scores['ssim'] = get_image_similarity_ssim(img1, img2)
                
                if "Feature Matching (ORB)" in comparison_methods:
                    scores['orb'] = get_image_similarity_orb(img1, img2)
                
                if "Color Histogram" in comparison_methods:
                    scores['histogram'] = get_image_similarity_histogram(img1, img2)
                
                if "Text Similarity (OCR)" in comparison_methods:
                    text1 = extract_text(img1, models)
                    text2 = extract_text(img2, models)
                    scores['text'] = get_text_similarity(text1, text2, models['text_model'])
                
                # Display results
                st.subheader("Comparison Results:")
                
                # Create columns for metrics based on selected methods
                cols = st.columns(len(comparison_methods))
                
                for i, method in enumerate(comparison_methods):
                    with cols[i]:
                        if method == "Structural Similarity (SSIM)":
                            st.metric("SSIM Score", f"{scores['ssim']:.2%}")
                        elif method == "Feature Matching (ORB)":
                            st.metric("ORB Score", f"{scores['orb']:.2%}")
                        elif method == "Color Histogram":
                            st.metric("Histogram Score", f"{scores['histogram']:.2%}")
                        elif method == "Text Similarity (OCR)":
                            st.metric("Text Score", f"{scores['text']:.2%}")
                
                # Calculate weighted average
                combined_score = sum(scores[k] * weights[k] for k in scores.keys())
                st.metric("Overall Similarity Score", f"{combined_score:.2%}")
                
                # Display extracted text if OCR was used
                if "Text Similarity (OCR)" in comparison_methods:
                    st.subheader("Extracted Text:")
                    text_col1, text_col2 = st.columns(2)
                    with text_col1:
                        st.text_area("Text from Image 1:", text1, height=100)
                    with text_col2:
                        st.text_area("Text from Image 2:", text2, height=100)

if __name__ == "__main__":
    main()
