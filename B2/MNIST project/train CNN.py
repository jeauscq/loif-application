import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and Preprocess the CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define CNN Architecture
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test,
                                                                  y_test))

# Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plot Training and Validation Loss
plt.figure(figsize=(14, 6), dpi=100)  # Increase figure size and resolution
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], color='blue', linestyle='-', linewidth=2,
         label='Training Loss')
plt.plot(history.history['val_loss'], color='red', linestyle='--', linewidth=2,
         label='Validation Loss')
plt.title('Loss Curve', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.grid(True)

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], color='blue', linestyle='-',
         linewidth=2, label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='red', linestyle='--',
         linewidth=2, label='Validation Accuracy')
plt.title('Accuracy Curve', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_validation_curves.png', dpi=300)  # Save in HD
plt.show()
