# 🧠 Custom 5-Digit MNIST Classification with CNN

This project focuses on generating a custom dataset of **5-digit sequences** using the classic **MNIST** handwritten digits dataset. The goal is to train a **Convolutional Neural Network (CNN)** to accurately classify these multi-digit images and evaluate its performance using both automated and manual methods.

---

## 📌 Project Overview

### 🔧 Step-by-step pipeline:

1. **Dataset Generation**
   - `Generate_Dataset.py` creates a new dataset by stitching together random 5-digit sequences from MNIST digit images.
   - The generated images are saved and labeled accordingly for supervised learning.

2. **Model Architecture**
   - `CNN.py` defines a custom CNN model designed for multi-digit image classification.
   - The model handles image preprocessing, convolutional layers, and output layers adapted to the 5-digit sequence task.

3. **Model Training**
   - `Train.py` loads the generated dataset and trains the CNN.
   - Training results, including metrics, are saved to `training_results.pkl`, and model weights are stored in `weights.pth`.

4. **Evaluation**
   - `Evaluation.py` evaluates the trained model on the test dataset.
   - Evaluation includes accuracy per digit position and total sequence accuracy.

5. **Manual Interface (Streamlit)**
   - `streamlit.py` allows manual digit-sequence classification through a user-friendly web app.
   - Users can upload their own 5-digit images for prediction and see the model's output live.

6. **Visualization**
   - `Visualization.ipynb` includes data and model visualization such as confusion matrices, prediction distributions, and training curves.
   - Architecture images (`nn.svg`, `converted_image.png`) visually explain the CNN structure and sample input format.

---

## 🚀 How to Run the Streamlit App

Make sure you have Python installed and the required packages (install manually).

### Install dependencies

```bash
pip install streamlit torch torchvision matplotlib
```
Then from the main repo directory run:

```bash
streamlit run streamlit.py
```
