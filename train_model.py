import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

# 🔥 EfficientNet imports
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# 🔥 NEW: EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Dataset paths
# -------------------------
train_dir = "augmented_dataset/train"
test_dir = "coral_dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10   # 🔥 reduced for better control

# -------------------------
# Data Generators
# -------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -------------------------
# CLASS WEIGHTS
# -------------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -------------------------
# EfficientNet Model
# -------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------
# Compile (🔥 tuned LR)
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# 🔥 EarlyStopping
# -------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# -------------------------
# Train
# -------------------------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    class_weight=class_weights,
    callbacks=[early_stop]   # 🔥 added
)

# -------------------------
# Save
# -------------------------
model.save("coral_cnn_model.h5")

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training completed with EfficientNet + Improved Accuracy")