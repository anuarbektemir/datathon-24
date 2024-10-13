import streamlit as st
from paddleocr import PaddleOCR
import re
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True)  # Set 'en' for English

# Compile regex pattern for number extraction
number_pattern = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:,[0-9]+)?')

# Optimized filter and calculate total area function
# Optimized filter and calculate total area function
def filter_and_calculate_total_area(items):
    filtered_items = []
    
    for item in items:
        # Replace commas with dots and handle duplicate dots
        item = item.replace('..', '.').replace(',', '.')
        
        # Extract value inside parentheses if present
        item = re.sub(r'.*\(([^)]+)\).*', r'\1', item)
        
        # Check if the item contains a valid number using regex
        match = number_pattern.fullmatch(item.strip())
        if match:
            filtered_items.append(item)
    
    # Calculate the total area, ensure no invalid items
    total_area = sum(map(float, filtered_items)) if filtered_items else 0.0
    
    return total_area, filtered_items


# Streamlit app
st.title("Разметка-классификация планировок")

# Upload multiple images
uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Initialize a list to store data for export
export_data = []

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
            st.image(image, caption=f"Изображение: {uploaded_file.name}", use_column_width=True)
            
            # Perform OCR on the uploaded image
            ocr_results = ocr_model.ocr(img_bgr)
            
            # Extract text
            text = [word_info[1][0] for line in ocr_results for word_info in line]
            
            # Calculate the total area
            total_area, filtered_items = filter_and_calculate_total_area(text)
            
            # Display the OCR results below the image
            st.write(f"Общая площадь: {total_area:.2f}")
            st.write(f"Площади индивидуальных комнат: {filtered_items}")
            st.write("---")
            
            # Store the data for export
            export_data.append({
                "Image Name": uploaded_file.name,
                "Total Area": total_area,
                "Filtered Items": ', '.join(filtered_items)
            })
        
        # Move to the next column
        current_col += 1
        # If the current column exceeds the number of columns, reset to the first column
        if current_col == num_columns:
            current_col = 0
            cols = st.columns(num_columns)  # Create a new set of columns for the next row

    # Convert the collected data to a DataFrame
    df_export = pd.DataFrame(export_data)

    # Display the Export button
    st.write("### Скачать результаты в Excel")
    if not df_export.empty:
        # Function to convert DataFrame to Excel and return it as BytesIO object
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            processed_data = output.getvalue()
            return processed_data

        # Call the function to get the excel data
        excel_data = convert_df_to_excel(df_export)
        
        # Create a download button
        st.download_button(
            label="Скачать в Excel",
            data=excel_data,
            file_name="image_ocr_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
