import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = "coral_dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

model = tf.keras.models.load_model("coral_cnn_model.h5")

loss, accuracy = model.evaluate(test_data)

print("✅ Test Accuracy:", accuracy * 100, "%")