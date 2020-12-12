import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180

def mkmodel(data_dir, save_path):
    #data_dir = "dataset/"

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

    def runmodel(arr):
        predictions = model.predict(tf.expand_dims(arr, 0))
        score = tf.nn.softmax(predictions[0])
        #print("Frame: {} FPS: {:.02} Class: {} Confidence: {}%".format(frameno, 1.0 / max(0.000001, time.time() - start), class_names[np.argmax(score)], 100 * np.max(score)))
        return (class_names[np.argmax(score)], 100 * np.max(score))

    model.save(save_path)

    return runmodel
    #return (class_names, model)

ident_model = mkmodel("datasets/game_ident", "models/ident.h5")
warzone_model = mkmodel("datasets/warzone", "models/warzone.h5")
warships_model = mkmodel("datasets/warships", "models/warships.h5")
overwatch_model = mkmodel("datasets/overwatch", "models/overwatch.h5")

#ident_model.save("models/ident.h5")
#warzone_model.save("models/warzone.h5")

# Now process a video file
"""import numpy as np
import cv2

cap = cv2.VideoCapture('test-1605829412.flv')

frameno = 0
while cap.isOpened():
    frameno += 1
    start = time.time()
    ret, frame = cap.read()
    if frameno % (60*10) != 0:
        # Only pull every 10s
        continue
    im = PIL.Image.fromarray(frame)
    im = im.resize((180, 180))
    arr = np.array(im)
    (pred_game, _) = ident_model(arr)
    (pred, score) = warzone_model(arr)
    print("Frame: {} FPS: {:.02} Game: {} Class: {} Confidence: {}%".format(frameno, 1.0 / max(0.000001, time.time() - start), pred_game, pred, score))
    #predictions = warzone_model.predict(tf.expand_dims(arr, 0))
    #score = tf.nn.softmax(predictions[0])
    #print("Frame: {} FPS: {:.02} Class: {} Confidence: {}%".format(frameno, 1.0 / max(0.000001, time.time() - start), class_names[np.argmax(score)], 100 * np.max(score)))
    pass

cap.release()"""
