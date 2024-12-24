import boto3
import sys
import argparse

def extract_text_with_textract(image_path):
    """Extract all text from an image using AWS Textract."""
    # Initialize Textract client
    textract = boto3.client('textract')
    print("Inside function, image_path:", image_path)
    
    # Read image file
    with open(image_path, 'rb') as image:
        print(image_path)
        image_bytes = image.read()
    
    # Call Textract
    response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    
    # Extract text and confidence scores for all detected text
    text_with_confidence = [
        {
            'text': item['Text'],
            'confidence': item['Confidence']
        }
        for item in response['Blocks'] 
        if item['BlockType'] == 'LINE'  # You could also use 'WORD' for word-level extraction
    ]
    
    return text_with_confidence

if __name__ == "__main__":
    try:
        print("Inside main")
        extracted_text = extract_text_with_textract("images/image5.jpeg")
        print("Extracted text with confidence scores:")
        for item in extracted_text:
            print(f"Text: {item['text']} (Confidence: {item['confidence']:.2f}%)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)