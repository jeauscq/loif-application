from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


def display_image_pair(image1, image2, similarity):
    """
    Display a window with the two images being compared with the percentage
    of similarity
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = axes
    ax1.imshow(image1)
    ax1.set_title('Image 1')
    ax1.axis('off')

    ax2.imshow(image2)
    ax2.set_title('Image 2')
    ax2.axis('off')

    fig.suptitle(f"The percentage of similarity is {similarity[0][0]}%", fontsize=16)
    plt.show()


# Load and Preprocess the MNIST Dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Load and Preprocess the CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0  # Normalizes the dataset
x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension (28, 28, 1)
x_test = np.expand_dims(x_test, axis=0)   # Add batch dimension (1, 28, 28, 1)

# Load the trained model
# model = load_model('mnist_cnn_model')
model1 = load_model('cifar10_cnn_model')
model2 = load_model('cifar10_cnn_model')

# Generate multiple cases
num_cases = 25  # Number of random pairs to display
for _ in range(num_cases):
    # Select random indices for image pairs
    idx1, idx2 = random.sample(range(len(x_test)), 2)
    image1 = x_test[:, idx1]
    image2 = x_test[:, idx2]
    # Make a prediction
    prediction1 = model1.predict(image1)
    prediction2 = model2.predict(image2)
    # Choose the most likely class
    predicted_class1 = np.argmax(prediction1, axis=1)
    predicted_class2 = np.argmax(prediction2, axis=1)
    # Obtain the cross probability
    cross_prediction1 = prediction1[:, predicted_class2]
    cross_prediction2 = prediction2[:, predicted_class1]
    # Calculates the similarity
    similarity = 100*0.5*(cross_prediction1+cross_prediction2)
    # Display the images with similarity percentage
    display_image_pair(image1[0, :, :, :, 0], image2[0, :, :, :, 0],
                       similarity)
