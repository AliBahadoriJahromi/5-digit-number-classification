import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import warnings

from CNN import CNN_Model
from Generate_Dataset import load_mnist_images, load_mnist_labels

warnings.filterwarnings("ignore")

class MNIST_Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]



if __name__ == '__main__':

    seed = 42  
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_images_path = 'src/data/train-images.idx3-ubyte'
    train_labels_path = 'src/data/train-labels.idx1-ubyte'
    test_images_path = 'src/data/t10k-images.idx3-ubyte'
    test_labels_path = 'src/data/t10k-labels.idx1-ubyte'

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    train_images = train_images / 255
    test_images = test_images / 255
    train_images = train_images.reshape(-1, 1, 28, 28).astype(np.float32)
    test_images = test_images.reshape(-1, 1, 28, 28).astype(np.float32)

    train_dataset = MNIST_Dataset(train_images, train_labels)
    test_dataset = MNIST_Dataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNN_Model()

    criterien = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    train_losses = []
    eval_losses = []
    train_corrects = []
    eval_corrects = []

    best_accuracy = 0.0  # Track the best accuracy
    best_model_state = None  # Store the state of the best model

    for i in range(epochs):
        model.train()
        correct = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterien(outputs, labels)

            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {i+1}  Train Loss: {loss}')
        train_losses.append(loss.item())
        train_corrects.append(correct)

        model.eval()
        correct = 0
        total = 0 
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterien(outputs, labels)

                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            eval_losses.append(loss.item())
            eval_corrects.append(correct)  

            accuracy = correct / total

            # Save the model if the accuracy improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict()  # Save the state of the best model

    # Save the best model after all epochs
    if best_model_state:
        torch.save(best_model_state, 'src/weights.pth')
        print("Best model saved with accuracy:", best_accuracy)


    with open('src/training_results.pkl', 'wb') as f:
        pickle.dump([train_losses, eval_losses, train_corrects, eval_corrects], f)

    




