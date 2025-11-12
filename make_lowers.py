#Image destroyer
# - HOW TO USE? - put normal images of faces into folder that called MainFaces and then activate this code 
#  After that you will have all images lowered to the LoweredFaces
#IMPORTANT NOTЕ:  Images would be deleted from MainFaces so make copies :)

import cv2
import os
import glob
import shutil


src_folder = "/Users/Nazar/Downloads/CV2/MainFaces"
dst_folder = "/Users/Nazar/Downloads/CV2/LoweredFaces"


target_size = (96, 128)
downscale_size = (24, 32)
jpeg_quality = 70

os.makedirs(dst_folder, exist_ok=True)


extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")


files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(src_folder, ext)))

print(f"Found: {len(files)}")


for path in files:
    try:
        filename = os.path.basename(path)
        dst_path = os.path.join(dst_folder, filename)

        img = cv2.imread(path)
        if img is None:
            print(f"Can't read: {filename}")
            continue

        # Уменьшаем, блюрим, делаем артефакты JPEG
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        small = cv2.resize(blur, downscale_size, interpolation=cv2.INTER_AREA)

        tmp_path = "/tmp/tmp_lr.jpg"
        cv2.imwrite(tmp_path, small, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        small = cv2.imread(tmp_path)
        low = cv2.resize(small, target_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(dst_path, low)

        # Удаляем оригинал
        os.remove(path)

        print(f"Saved!: {filename}")

    except Exception as e:
        print(f"Error with path: {path}: {e}")

print("All images redacted!")
