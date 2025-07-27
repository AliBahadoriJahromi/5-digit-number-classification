import struct
import numpy as np
import torch
from loguru import logger


def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
def create_five_digit_dataset(images, labels, num_samples=10000):
    five_digit_images = []
    five_digit_labels = []
    num_total_samples = len(labels)

    for _ in range(num_samples):
        # انتخاب اولین عدد (که نباید صفر باشد)
        non_zero_indices = np.where(labels != 0)[0]
        first_index = np.random.choice(non_zero_indices, 1)
        first_digit = labels[first_index]
        first_image = images[first_index]

        # انتخاب چهار عدد بعدی به صورت تصادفی (بدون محدودیت)
        remaining_indices = np.random.choice(num_total_samples, 4, replace=False)
        remaining_digits = labels[remaining_indices]
        remaining_images = images[remaining_indices]

        # ترکیب تصاویر
        combined_image = np.hstack([first_image[0], *remaining_images])

        # ایجاد برچسب برای عدد پنج‌رقمی
        combined_label = int(''.join(map(str, [first_digit[0], *remaining_digits])))

        # ذخیره تصویر و برچسب
        five_digit_images.append(combined_image)
        five_digit_labels.append(combined_label)

    return np.array(five_digit_images), np.array(five_digit_labels)
    

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

    num_samples = 20000
    five_digit_images, five_digit_labels = create_five_digit_dataset(train_images, train_labels, num_samples=num_samples)

    np.save('src/data/five_digit_train_images.npy', five_digit_images)
    np.save('src/data/five_digit_train_labels.npy', five_digit_labels)   
    logger.info("Dataset generated successfuly!") 

