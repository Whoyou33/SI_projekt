import pyautogui
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from PIL import ImageGrab
import time
import pyautogui


image_size = (32, 32)
batch_size = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "DataSet/Train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "DataSet/Train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

import matplotlib.pyplot as plt
## nadawanie etykiet
plt.figure(figsize=(32, 32))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
## przeksztalcanie obrazow (nam niepotrzebne)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
plt.figure(figsize=(32, 32))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
##dostarcza partie rozszerzonych obrazów
augmented_train_ds = train_ds.map(
     lambda x, y: (data_augmentation(x, training=True), y))

##opcja druga
#inputs = keras.Input(shape=input_shape)
#x = data_augmentation(inputs)
#x = layers.Rescaling(1. / 255)(x)
#...  # Rest of the model


##buforowanie wstepne
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    print(inputs)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

####uczenie
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
##############################################
#img = keras.preprocessing.image.load_img(
#    "DataSet/Train/1/gold_15.png", target_size=image_size
#)

screenWidth, screenHeight = pyautogui.size()
##plecak (1286,206)

b=1
while b >= 1:
    ### pierwsza krataka
    img = ImageGrab.grab(bbox=(1200,394,1232,426))
    img_array = keras.preprocessing.im2age.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent no gold and %.2f percent gold."
        % (100 * (1 - score), 100 * score)
    )
    if 100*score >= 51:
        currentMouseX, currentMouseY = pyautogui.position()
        pyautogui.moveTo(1215, 404)
        pyautogui.drag(71, -198, 0.5, button='left')
        pyautogui.press('enter')
    time.sleep(0.5)

    ###druga kratka
    img = ImageGrab.grab(bbox=(1237, 394, 1269, 426))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent no gold and %.2f percent gold."
        % (100 * (1 - score), 100 * score)
    )
    if 100*score >= 51:
        currentMouseX, currentMouseY = pyautogui.position()
        pyautogui.moveTo(1250, 411)
        pyautogui.drag(36, -198, 0.5, button='left')
        pyautogui.press('enter')

    time.sleep(0.5)
    ###trzecia kratka
    img = ImageGrab.grab(bbox=(1274, 394, 1306, 426))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent no gold and %.2f percent gold."
        % (100 * (1 - score), 100 * score)
    )
    if 100*score >= 51:
        currentMouseX, currentMouseY = pyautogui.position()
        pyautogui.moveTo(1285, 404)
        pyautogui.drag(1, -198, 0.5, button='left')
        pyautogui.press('enter')

    time.sleep(0.5)
    ###czwarta kratka
    img = ImageGrab.grab(bbox=(1311,394,1343,426))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent no gold and %.2f percent gold."
        % (100 * (1 - score), 100 * score)
    )
    if 100*score >= 51:
        currentMouseX, currentMouseY = pyautogui.position()
        pyautogui.moveTo(1320, 404)
        pyautogui.drag(-34, -198, 0.5, button='left')
        pyautogui.press('enter')

    time.sleep(0.5)
    ### zamykanie oknad
    #pyautogui.moveTo(1355,382)
    #pyautogui.click()
#img = ImageGrab.grab(bbox=(1200,394,1232,426))