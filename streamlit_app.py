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
st.title("Multi-Image OCR with Gallery View and Area Calculation")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Define how many images to show per row (number of columns)
    num_columns = 3

    # Initialize index
    current_col = 0
    cols = st.columns(num_columns)  # Create the columns initially
    
    for uploaded_file in uploaded_files:
        # Open the uploaded image with PIL and convert it to a format suitable for OpenCV
        image = Image.open(uploaded_file)
        
        # Convert the image to RGB if it's not in that mode (OCR requires RGB input)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to OpenCV format (numpy array)
        img_array = np.array(image)

        # Convert the RGB numpy array to BGR format for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Display the image in the appropriate column
        with cols[current_col]:
            st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)
            
            # Perform OCR on the uploaded image
            ocr_results = ocr_model.ocr(img_bgr)
            
            # Extract text
            text = [word_info[1][0] for line in ocr_results for word_info in line]
            
            # Calculate the total area
            total_area, filtered_items = filter_and_calculate_total_area(text)
            
            # Display the OCR results below the image
            st.write(f"Total Area: {total_area}")
            st.write(f"Filtered Items: {filtered_items}")
            st.write("---")
        
        # Move to the next column
        current_col += 1
        # If the current column exceeds the number of columns, reset to the first column
        if current_col == num_columns:
            current_col = 0
            cols = st.columns(num_columns)  # Create a new set of columns for the next row