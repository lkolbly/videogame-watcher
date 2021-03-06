#import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import time
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180

def mkmodel(data_dir, save_path):
    # Load the dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    # Plot the dataset
    """plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()"""

    # Build the model
    num_classes = len(class_names)
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()

    # Train the model
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Visualize the results
    """acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()"""

    model.save(save_path)

    with open(f"{save_path}.json", "w") as f:
        f.write(json.dumps(class_names))

def main():
    if len(sys.argv) != 3:
        print(f"Expected usage: {sys.argv[0]} <Path to datasets> <Output model path>")
        print("You can generate the datasets directory using the mkdatasets.py script")
        return

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]

    mkmodel(f"{dataset_path}/game_ident", f"{model_path}/ident.h5")
    mkmodel(f"{dataset_path}/warzone", f"{model_path}/warzone.h5")
    mkmodel(f"{dataset_path}/warships", f"{model_path}/warships.h5")
    mkmodel(f"{dataset_path}/overwatch", f"{model_path}/overwatch.h5")

if __name__ == "__main__":
    main()
