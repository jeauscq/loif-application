from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, Activation
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras import backend as K
from keras.optimizers import Adam

import tensorflow as tf

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the images to [0, 1] range and expand the dimensions to include a channel (grayscale)
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

metric = 'Mean Absolute Error (MAE)'
metric = 'Mean Squared Error (MSE)'
metric = 'Structural Similarity Index (SSIM)'
metric = "Average"

loaded_size = 70000


def load_training_data():
    # read by default 1st sheet of an excel file
    df = pd.read_excel('train_image_comparison_normalized.xlsx')

    # First let's separate the dataset from 1 matrix to a list of matricies
    image_list = x_train[:loaded_size]
    # label_list = y_train[:1000]

    left_input = []
    right_input = []
    targets = []

    df = df.reset_index()  # make sure indexes pair with number of rows
    for _, row in df.iterrows():
        left_input.append(image_list[int(row['Index 1'])])
        right_input.append(image_list[int(row['Index 2'])])
        targets.append(float(row[metric]))

    left_input = np.squeeze(np.array(left_input))
    right_input = np.squeeze(np.array(right_input))
    targets = np.squeeze(np.array(targets))

    return [left_input, right_input, targets]


def load_test_data():
    # read by default 1st sheet of an excel file
    df = pd.read_excel('test_image_comparison_normalized.xlsx')

    # First let's separate the dataset from 1 matrix to a list of matricies
    image_list = x_test[:loaded_size]
    # label_list = y_train[:1000]

    left_input = []
    right_input = []
    targets = []

    df = df.reset_index()  # make sure indexes pair with number of rows
    for _, row in df.iterrows():
        left_input.append(image_list[int(row['Index 1'])])
        right_input.append(image_list[int(row['Index 2'])])
        targets.append(float(row[metric]))

    left_input = np.squeeze(np.array(left_input))
    right_input = np.squeeze(np.array(right_input))
    targets = np.squeeze(np.array(targets))

    return [left_input, right_input, targets]


# Plot training results
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


def define_siamese():
    # We have 2 inputs, 1 for each picture
    left_input = Input((28, 28, 1))
    right_input = Input((28, 28, 1))

    # We will use 2 instances of 1 network for this task
    convnet = Sequential([
        Conv2D(5, 3, input_shape=(28, 28, 1)),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(5, 3),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(7, 2),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(7, 2),
        Activation('relu'),
        Flatten(),
        Dense(18),
        Activation('sigmoid')
    ])
    # Connect each 'leg' of the network to each input
    # Remember, they have the same weights
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # Getting the L1 Distance between the 2 encodings
    L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # Define a learning rate schedule
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=75,  # Adjust this based on your training steps
        decay_rate=0.2,   # Adjust this based on how quickly you want the learning rate to decay
        staircase=True     # If True, decay the learning rate at discrete intervals
    )

    # Create the Adam optimizer with the learning rate schedule
    optimizer = Adam(lr_schedule)

    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer,
                        metrics=['accuracy'])
    return siamese_net


if __name__ == "__main__":
    siamese_net = define_siamese()
    left_input, right_input, targets = load_training_data()
    test_left, test_right, test_targets = load_test_data()

    siamese_net.summary()
    history = siamese_net.fit([left_input, right_input], targets,
                              batch_size=200, epochs=12, verbose=1,
                              validation_data=([test_left, test_right],
                                               test_targets))

    plot_training_history(history)

    predictions = siamese_net.predict([test_left[:10], test_right[:10]])

    # Print the results
    for i in range(10):
        print(f"Prediction: {predictions[i][0]:.4f}, Actual Target: {test_targets[i]}")
