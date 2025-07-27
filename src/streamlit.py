import streamlit as st
import torch
import numpy as np
from PIL import Image

from CNN import CNN_Model


# Load your pre-trained model here
model = CNN_Model()
model.load_state_dict(torch.load("src/weights.pth"))
model.eval()

def predict_single_image(model, image_path):
    # Load the image using PIL and convert it to grayscale
    img = Image.open(image_path).convert('L')  # Grayscale conversion
    img = np.array(img)  # Convert image to numpy array

    # Ensure the image has the correct shape (28, 140)
    if img.shape != (28, 140):
        raise ValueError(f"Image must have shape (28, 140), but got {img.shape}")

    # Normalize the image
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # Convert the image to a PyTorch tensor and add channel and batch dimensions
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 140)

    # Split the image into 5 sub-images (28x28)
    sub_images = [img_tensor[:, :, :, i * 28:(i + 1) * 28] for i in range(5)]

    predicted_number = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for sub_img in sub_images:
            output = model(sub_img)  # Forward pass through the model
            _, predicted_digit = torch.max(output, 1)  # Get the predicted digit (class index)
            predicted_number = predicted_number * 10 + predicted_digit.item()

    return predicted_number

# Streamlit interface
st.title("5-Digit Number Prediction from Image")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    # Save the uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the number from the image
    try:
        predicted_number = predict_single_image(model, "temp_image.png")
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        st.markdown(f"<h2>Predicted 5-digit number: <span style='color: green;'>{predicted_number}</span></h2>", unsafe_allow_html=True)
        # st.write(f"Predicted 5-digit number: {predicted_number}")
    except ValueError as e:
        st.error(str(e))
