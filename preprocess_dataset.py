import os
import cv2
import numpy as np

input_dir = "coral_dataset/train"
output_dir = "processed_dataset/train"

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    save_path = os.path.join(output_dir, class_name)
    os.makedirs(save_path, exist_ok=True)

    print("Processing:", class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        # 1. Resize
        img = cv2.resize(img, (224, 224))

        # 2. Noise removal
        img = cv2.medianBlur(img, 3)
        img = cv2.GaussianBlur(img, (5,5), 0)

        # 3. Contrast enhancement (CLAHE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        img = cv2.merge((l,a,b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        # 4. Edge enhancement
        edges = cv2.Canny(img, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)

        # 5. Sharpening
        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)

        # 6. Normalization
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Save
        save_file = os.path.join(save_path, img_name)
        cv2.imwrite(save_file, img)

print("✅ Preprocessing completed!")