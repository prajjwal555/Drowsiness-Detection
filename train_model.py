import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_DIR = "data/train"
MODEL_DIR = "models"
RESULTS_DIR = "results"

IMG_SIZE = (24, 24)
BATCH_SIZE = 64
EPOCHS = 15

SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# TRAINING DATA
# Augmentation is applied ONLY to training images
# =========================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    validation_split=0.15,

    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    horizontal_flip=True
)


# =========================================================
# VALIDATION DATA
# NO augmentation
# Only normalization is applied
# =========================================================

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.15
)


# =========================================================
# TRAINING GENERATOR
# =========================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",

    batch_size=BATCH_SIZE,

    subset="training",

    shuffle=True,
    seed=SEED
)


# =========================================================
# VALIDATION GENERATOR
# =========================================================

validation_generator = validation_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",

    batch_size=BATCH_SIZE,

    subset="validation",

    shuffle=False,
    seed=SEED
)


# =========================================================
# DATASET INFORMATION
# =========================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nClass mapping:")
print(train_generator.class_indices)

print(
    "\nTraining images:",
    train_generator.samples
)

print(
    "Validation images:",
    validation_generator.samples
)

print(
    "Number of classes:",
    train_generator.num_classes
)


# =========================================================
# CNN ARCHITECTURE
# =========================================================

model = Sequential([

    # Input
    Input(shape=(24, 24, 1)),


    # -----------------------------------------------------
    # Convolution Block 1
    # -----------------------------------------------------

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.25),


    # -----------------------------------------------------
    # Convolution Block 2
    # -----------------------------------------------------

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.25),


    # -----------------------------------------------------
    # Convolution Block 3
    # -----------------------------------------------------

    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.30),


    # -----------------------------------------------------
    # Classification Head
    # -----------------------------------------------------

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.50),

    Dense(
        2,
        activation="softmax"
    )
])


# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "drowsiness_cnn_final.keras"
)


# Save the model with the BEST validation accuracy
checkpoint = ModelCheckpoint(
    best_model_path,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


# Stop when validation accuracy stops improving
early_stopping = EarlyStopping(
    monitor="val_accuracy",

    mode="max",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


# Reduce learning rate when validation accuracy plateaus
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",

    mode="max",

    factor=0.5,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60 + "\n")


history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    "\nBest model saved to:",
    best_model_path
)


# =========================================================
# TRAINING ACCURACY CURVE
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    RESULTS_DIR,
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# TRAINING LOSS CURVE
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    RESULTS_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# FINAL INFORMATION
# =========================================================

print("\nTraining curves saved:")

print(
    " - results/accuracy_curve.png"
)

print(
    " - results/loss_curve.png"
)

print("\nDone!")