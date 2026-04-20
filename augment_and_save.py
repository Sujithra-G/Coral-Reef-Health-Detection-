import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ✅ INPUT = processed dataset (IMPORTANT)
input_dir = "processed_dataset/train"

# ✅ OUTPUT = new augmented dataset (separate)
output_dir = "augmented_dataset/train"

os.makedirs(output_dir, exist_ok=True)

# ✅ Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.3,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.7, 1.3]
)

# Loop through each class
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    save_path = os.path.join(output_dir, class_name)

    os.makedirs(save_path, exist_ok=True)

    print("Augmenting:", class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Load image
        img = load_img(img_path, target_size=(224,224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0

        for batch in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=save_path,
            save_prefix="aug",
            save_format="png"
        ):
            i += 1

            # 🔥 More augmentation for Dead class
            if class_name == "Dead":
                limit = 8
            else:
                limit = 3

            if i >= limit:
                break

print("✅ Augmented dataset created successfully!")