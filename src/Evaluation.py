import torch
import numpy as np
from CNN import CNN_Model


def evaluate_model(model, images, labels):
    # Normalizing Data
    images = images / 255
    images = images.reshape(-1, 1, 28, 140).astype(np.float32)

    # Convert images and labels from numpy arrays to PyTorch tensors
    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    all_preds = []
    all_labels = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Loop through each image
        for img, label in zip(images_tensor, labels_tensor):
            predicted_digits = []

            # Split the image into 5 sub-images (28, 28)
            sub_images = [img[:, :, i * 28:(i + 1) * 28] for i in range(5)]

            # Predict each digit using the model
            predicted_number = 0
            for sub_img in sub_images:
                sub_img = sub_img.unsqueeze(0)  # Add batch and channel dimensions
                output = model(sub_img)
                _, predicted_digit = torch.max(output, 1)  # Get the predicted class (digit)
                predicted_number = predicted_number * 10 + predicted_digit.item()
            # Append the predicted 5-digit number and true label
            all_preds.append(predicted_number)
            all_labels.append(label.item())  # Store label as integer

    return all_preds, all_labels

def calculate_accuracy(predictions, true_labels):
    correct_count = sum([pred == true for pred, true in zip(predictions, true_labels)])
    total_count = len(true_labels)
    accuracy = (correct_count / total_count) * 100
    return accuracy


if __name__ == '__main__':
    five_digit_images_path = 'src/data/five_digit_train_images.npy'
    five_digit_labels_path = 'src/data/five_digit_train_labels.npy'

    images = np.load(five_digit_images_path)
    labels = np.load(five_digit_labels_path)

    model = CNN_Model()
    model.load_state_dict(torch.load('src/weights.pth'))
    model.eval()

    predictions, true_labels = evaluate_model(model, images, labels)

    accuracy = calculate_accuracy(predictions, true_labels)
    print(f"Accuracy: {accuracy:.2f}%")