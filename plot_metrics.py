import pickle
import matplotlib.pyplot as plt

# -------------------------
# Load training history
# -------------------------
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# -------------------------
# Accuracy Plot
# -------------------------
plt.figure()
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig("accuracy_curve.png", dpi=300, bbox_inches="tight")   # SAVE IMAGE
plt.show()

# -------------------------
# Loss Plot
# -------------------------
plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")   # SAVE IMAGE
plt.show()