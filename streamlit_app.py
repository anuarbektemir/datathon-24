import streamlit as st
from paddleocr import PaddleOCR
import re
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    clf_norm = pickle.load(f)

# Maximum length for padding input array
max_length = 17

# Function to predict the number of rooms
def predict_rooms(input_array):
    input_array_sorted = sorted(input_array, reverse=True)
    total_area = sum(input_array)
    input_array_normalized = [x / total_area for x in input_array_sorted]
    input_array_padded = np.pad(input_array_normalized, (0, max_length - len(input_array_normalized)), 'constant')
    
    # Predict the class (number of rooms)
    prediction = clf_norm.predict([input_array_padded])
    
    # Predict probabilities
    prob = clf_norm.predict_proba([input_array_padded])
    
    # Create a dictionary with room-type probabilities
    class_probs = {f'{i+1}-комнатная': prob[0][i] for i in range(len(prob[0]))}
    
    return prediction[0], class_probs

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True)

# Regex pattern for number extraction
number_pattern = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:,[0-9]+)?')

# Function to filter and calculate total area
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
            try:
                # Skip the item if it doesn't contain a decimal point
                if '.' not in item:
                    continue
                
                # Convert to float
                value = float(item)
                
                # Ignore values bigger than 200 or if the number is not positive
                if 0 < value <= 200:
                    filtered_items.append(item)
            except ValueError:
                # Ignore items that cannot be converted to float
                continue
    
    # Calculate the total area
    total_area = sum(map(float, filtered_items)) if filtered_items else 0.0
    
    return total_area, filtered_items

# Streamlit app
st.title("Разметка-классификация планировок")

# Upload multiple images
uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

# Initialize a list to store data for export
export_data = []

if uploaded_files:
    # Define how many images to show per row (number of columns)
    num_columns = 3

    # Split the uploaded files into groups for each row
    rows = [uploaded_files[i:i + num_columns] for i in range(0, len(uploaded_files), num_columns)]

    # Iterate over each row
    for row in rows:
        cols = st.columns(num_columns)  # Create the columns for the current row
        
        for col, uploaded_file in zip(cols, row):
            try:
                # Open the uploaded image with PIL and convert it to a format suitable for OpenCV
                image = Image.open(uploaded_file)
                
                # Convert the image to RGB if it's not in that mode (OCR requires RGB input)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Convert to OpenCV format (numpy array)
                img_array = np.array(image)

                # Display the image in the appropriate column
                with col:
                    st.image(image, caption=f"Изображение: {uploaded_file.name}", use_column_width=True)
                    
                    # Perform OCR on the uploaded image
                    ocr_results = ocr_model.ocr(img_array)
                    
                    # Extract text
                    text = [word_info[1][0] for line in ocr_results for word_info in line]
                    
                    # Calculate the total area
                    total_area, filtered_items = filter_and_calculate_total_area(text)
                    
                    # Display the OCR results below the image
                    st.write(f"Общая площадь: {total_area:.2f}")
                    st.write(f"Площади индивидуальных комнат: {filtered_items}")
                    st.write("---")
                    
                    example_input = [float(item) for item in filtered_items]
                    predicted_rooms, probabilities = predict_rooms(example_input)

                    st.write(f'Предсказано: {predicted_rooms}-комнатная квартира')
                    st.write('Вероятности:')
                    for room_type, prob in probabilities.items():
                        st.write(f'{room_type}: {prob:.2f}')

                    # Store the data for export
                    export_data.append({
                        "Image Name": uploaded_file.name,
                        "Total Area": total_area,
                        "Filtered Items": ', '.join(filtered_items),
                        "Predicted Rooms": predicted_rooms,
                        "Prediction probability": probabilities
                    })
            except:
                col.write("Не удалось распознать изображение")
    
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
