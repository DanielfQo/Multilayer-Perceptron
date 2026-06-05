import os
import cv2
import numpy as np
import struct

DATASET_PATH = r"simpsons-mnist\dataset\rgb"
IMG_SIZE = (64, 64)

# True = escala de grises
# False = RGB
GRAYSCALE = False

classes = sorted(os.listdir(os.path.join(DATASET_PATH, "train")))
class_to_label = {c: i for i, c in enumerate(classes)}

print("Clases:", class_to_label)

def save_split(split):
    samples = []

    split_path = os.path.join(DATASET_PATH, split)

    for class_name in classes:

        class_dir = os.path.join(split_path, class_name)

        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, file)

            if GRAYSCALE:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)

            img = img.astype(np.float32) / 255.0

            x = img.flatten()

            label = class_to_label[class_name]

            samples.append((x, label))

    channels = 1 if GRAYSCALE else 3
    n_features = IMG_SIZE[0] * IMG_SIZE[1] * channels

    output_file = f"{split}.bin"

    with open(output_file, "wb") as f:

        # Cabecera
        f.write(struct.pack("i", len(samples)))
        f.write(struct.pack("i", IMG_SIZE[0]))
        f.write(struct.pack("i", IMG_SIZE[1]))
        f.write(struct.pack("i", channels))

        # Datos
        for x, label in samples:
            f.write(struct.pack("i", label))
            f.write(x.astype(np.float32).tobytes())

    print(
        f"{output_file}: "
        f"{len(samples)} imágenes, "
        f"{channels} canal(es), "
        f"{n_features} características"
    )

save_split("train")
save_split("test")