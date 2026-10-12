# train.py

import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================

DATA_PATH = r"C:\Users\John Nathan Badon\Downloads\ResearchImage"

MODEL_SAVE_PATH = r"C:\Users\John Nathan Badon\Downloads\ResearchModel\FSL_ILY_best.h5"

FINAL_MODEL_PATH = r"C:\Users\John Nathan Badon\Downloads\ResearchModel\FSL_ILY.h5"


# =========================
# SETTINGS
# =========================

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20


# =========================
# LOAD DATASET
# =========================

classes = sorted(os.listdir(DATA_PATH))

noOfClasses = len(classes)

print("Classes detected:", noOfClasses)
print("Class names:", classes)


image_paths = []
class_labels = []


for index, folder in enumerate(classes):

    folder_path = os.path.join(DATA_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    for img in os.listdir(folder_path):

        if img.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            image_paths.append(
                os.path.join(folder_path, img)
            )

            class_labels.append(index)


print("Total images:", len(image_paths))


# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    class_labels,
    test_size=0.15,
    random_state=42,
    stratify=class_labels
)


X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)


print(
    "Train:",
    len(X_train),
    "Validation:",
    len(X_val),
    "Test:",
    len(X_test)
)



# =========================
# IMAGE LOADING
# =========================

def load_image(path, label):

    img = tf.io.read_file(path)

    img = tf.image.decode_image(
        img,
        channels=1,
        expand_animations=False
    )

    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE]
    )

    img = tf.cast(
        img,
        tf.float32
    ) / 255.0


    label = tf.one_hot(
        label,
        noOfClasses
    )

    return img, label



# =========================
# AUGMENTATION
# =========================

def augment_image(image, label):

    image = tf.image.random_brightness(
        image,
        max_delta=0.2
    )

    image = tf.image.random_contrast(
        image,
        lower=0.8,
        upper=1.2
    )


    image = tf.clip_by_value(
        image,
        0.0,
        1.0
    )

    return image, label



# =========================
# DATASET CREATION
# =========================

AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(paths, labels, augment=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )


    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )


    if augment:

        ds = ds.map(
            augment_image,
            num_parallel_calls=AUTOTUNE
        )

        ds = ds.shuffle(
            2000
        )


    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        AUTOTUNE
    )


    return ds



train_ds = make_dataset(
    X_train,
    y_train,
    True
)


val_ds = make_dataset(
    X_val,
    y_val,
    False
)


test_ds = make_dataset(
    X_test,
    y_test,
    False
)



# Check input shape

for images, labels in train_ds.take(1):

    print(
        "Input batch:",
        images.shape
    )



# =========================
# MODEL
# =========================

def create_model():

    model = Sequential([

        Input(
            shape=(64,64,1)
        ),


        Conv2D(
            32,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(),



        Conv2D(
            64,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(),



        Conv2D(
            128,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(),



        Conv2D(
            256,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(),



        Flatten(),


        Dense(
            256,
            activation="relu"
        ),

        Dropout(
            0.5
        ),


        Dense(
            noOfClasses,
            activation="softmax"
        )

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0008
        ),

        loss="categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model



model = create_model()

model.summary()



# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)



checkpoint = ModelCheckpoint(

    MODEL_SAVE_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"

)



reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=1e-6

)



# =========================
# TRAIN
# =========================

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]

)



# =========================
# TEST
# =========================

score = model.evaluate(
    test_ds
)


print(
    "Test accuracy:",
    score[1] * 100,
    "%"
)



# =========================
# SAVE FINAL MODEL
# =========================

model.save(
    FINAL_MODEL_PATH
)


print(
    "Saved:",
    FINAL_MODEL_PATH
)



# =========================
# TRAIN GRAPH
# =========================

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.legend()

plt.show()