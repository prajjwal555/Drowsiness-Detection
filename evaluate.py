import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "models/drowsiness_cnn_final.keras"
TEST_DIR = "data/test"
RESULTS_DIR = "results"

IMG_SIZE = (24, 24)
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Load improved model
# -----------------------------
print("Loading improved model...")

from tensorflow.keras.models import load_model

model = load_model(MODEL_PATH)

print("Improved model loaded successfully!")

# -----------------------------
# Load test dataset
# -----------------------------
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass mapping:")
print(test_generator.class_indices)

print(
    f"\nTotal test images: "
    f"{test_generator.samples}"
)


# -----------------------------
# Predictions
# -----------------------------
print("\nRunning predictions...")

probabilities = model.predict(
    test_generator,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

y_true = test_generator.classes


# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

try:
    auc = roc_auc_score(
        y_true,
        probabilities[:, 1]
    )
except Exception:
    auc = None


# -----------------------------
# Display results
# -----------------------------
print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(
    f"Accuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

if auc is not None:
    print(f"ROC-AUC   : {auc:.4f}")

print("=" * 50)


# -----------------------------
# Classification report
# -----------------------------
class_names = list(
    test_generator.class_indices.keys()
)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# Save report
report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:

    f.write(
        "Drowsiness Detection Model Evaluation\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f}\n"
    )

    if auc is not None:
        f.write(
            f"ROC-AUC   : {auc:.4f}\n"
        )

    f.write(
        "\nClassification Report:\n"
    )

    f.write(report)


# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Drowsiness Detection"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=45
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# Add numbers
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nConfusion matrix saved to: "
    f"{cm_path}"
)

print(
    f"Classification report saved to: "
    f"{report_path}"
)

print("\nEvaluation completed successfully!")