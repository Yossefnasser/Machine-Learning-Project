import mnist_reader
import os
import numpy as np
import matplotlib.pyplot as plt

my_cur_path = os.path.dirname(os.path.abspath(__file__))
print("my_cur_path: ", my_cur_path)
data_path = os.path.join(my_cur_path, 'dataset')
print("data_path: ", data_path)
X_train, y_train = mnist_reader.load_mnist(data_path, kind='train')
X_test, y_test   = mnist_reader.load_mnist(data_path, kind='t10k')

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print("Training data shape:", X_train.shape)  
print("Training labels shape:", y_train.shape) 
print("Test data shape:", X_test.shape) 
print("Test labels shape:", y_test.shape)
print(f"\nNumber of classes: {len(np.unique(y_train))}")
print(f"Classes: {sorted(np.unique(y_train))}")

# Fashion-MNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("\nClass distribution in training set:")
for i in range(10):
    count = np.sum(y_train == i)
    print(f"  Class {i} ({class_names[i]:15s}): {count:5d} samples ({count/len(y_train)*100:.1f}%)")

# Visualize sample images from each class
print("\n" + "="*60)
print("VISUALIZING SAMPLE IMAGES")
print("="*60)

# Create a grid showing one example from each class
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Images from Each Class (Training Set)', fontsize=14, fontweight='bold')

for i in range(10):
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    
    # Find first occurrence of class i
    idx = np.where(y_train == i)[0][0]
    
    # Reshape from 784 to 28x28 and display
    image = X_train[idx].reshape(28, 28)
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Class {i}: {class_names[i]}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Show random samples from training set
print("\nDisplaying 10 random samples from training set...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Random Sample Images from Training Set', fontsize=14, fontweight='bold')

random_indices = np.random.choice(len(X_train), 10, replace=False)
for i, idx in enumerate(random_indices):
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    
    image = X_train[idx].reshape(28, 28)
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {y_train[idx]} ({class_names[y_train[idx]]})', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Show some statistics
print("\n" + "="*60)
print("DATA STATISTICS")
print("="*60)
print(f"Pixel value range: [{X_train.min()}, {X_train.max()}]")
print(f"Mean pixel value: {X_train.mean():.2f}")
print(f"Std pixel value: {X_train.std():.2f}")
print(f"\nFirst 5 training labels: {y_train[:5]}")
print(f"First 5 test labels: {y_test[:5]}") 
