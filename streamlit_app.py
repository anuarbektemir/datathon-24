import streamlit as st
from paddleocr import PaddleOCR
import re
import cv2
import numpy as np
from PIL import Image

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True)  # Set 'en' for English

# Compile regex pattern for number extraction
number_pattern = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:,[0-9]+)?')

# Optimized filter and calculate total area function
def filter_and_calculate_total_area(items):
    filtered_items = []
    
    for item in items:
        # Replace commas with dots and handle duplicate dots
        item = item.replace('..', '.').replace(',', '.')
        
        # Extract value inside parentheses if present
        item = re.sub(r'.*\(([^)]+)\).*', r'\1', item)
        
        # Check if the item contains a valid number and filter
        if '.' in item:
            filtered_items.append(item)
    
    # Calculate the total area
    total_area = sum(map(float, filtered_items))
    
    return total_area, filtered_items

# Streamlit app
st.title("Image OCR and Area Calculation")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image with PIL and convert it to a format suitable for OpenCV
    image = Image.open(uploaded_file)
    img_array = np.array(image)  # Convert image to numpy array
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR format for OpenCV
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform OCR on the uploaded image
    ocr_results = ocr_model.ocr(img_bgr)
    
    # Extract text
    text = [word_info[1][0] for line in ocr_results for word_info in line]
    
    # Calculate the total area
    total_area, filtered_items = filter_and_calculate_total_area(text)
    
    # Display the results
    st.write(f"Total Area: {total_area}")
    st.write(f"Filtered Items: {filtered_items}")
