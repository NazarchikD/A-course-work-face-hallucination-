#Train by Nazar Domshenko
# This is the main part, you should run this first. This is our neouron, that would train us a Model 
# I used a Calebas Dataset i will put it on rep
# IMPORTANT NOTЕ: Put your path where you downloaded dataset, u can change the setting to train your model better, i used this parameters, but u can use whatever you want. :)


#Important imports
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Settings
HR_DIR = "/Users/Nazar/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"
MODEL_PATH = "face_hallucination.h5"
UPSCALE = 4
EPOCHS = 40        
BATCH = 8
TARGET_W, TARGET_H = 96, 128
MAX_IMAGES = 2000  


def make_dataset(hr_dir: str, scale: int = 4):
    print(f"[load] Reading img from: {hr_dir}", flush=True)
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"No folders have been found in: {hr_dir}")
    files = [f for f in os.listdir(hr_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"В {hr_dir} there's no .jpg/.png")

    if MAX_IMAGES is not None:
        files = files[:MAX_IMAGES]

    X_lr, X_hr = [], []
    for i, f in enumerate(files, 1):
        path = os.path.join(hr_dir, f)
        hr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if hr is None:
            continue
        hr = cv2.resize(hr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        lr_small = cv2.resize(hr, (TARGET_W//scale, TARGET_H//scale), interpolation=cv2.INTER_AREA)
        lr_up = cv2.resize(lr_small, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
        X_lr.append(lr_up[..., None] / 255.0)
        X_hr.append(hr[..., None] / 255.0)
        if i % 200 == 0:
            print(f"  ...Downloaded {i} / {len(files)}", flush=True)

    X_lr = np.asarray(X_lr, dtype=np.float32)
    X_hr = np.asarray(X_hr, dtype=np.float32)
    print(f"[load] Downloaded {len(X_lr)} images", flush=True)
    return X_lr, X_hr

def build_model():
    print("[model] build a model", flush=True)
    model = models.Sequential([
        layers.Conv2D(64, 9, activation='relu', padding='same', input_shape=(None, None, 1)),
        layers.Conv2D(32, 5, activation='relu', padding='same'),
        layers.Conv2D(1, 5, activation='sigmoid', padding='same')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    print("[model] Completed", flush=True)
    return model

#Neuron training
def train():
    print("TRAIN START...", flush=True)
    hr_dir = HR_DIR  
    X_lr, X_hr = make_dataset(hr_dir, UPSCALE)
    print(f"[shape] X_lr: {X_lr.shape}  X_hr: {X_hr.shape}", flush=True)

    model = build_model()
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH, monitor='val_loss',
        save_best_only=True, save_weights_only=False, verbose=1
    )
    model.fit(
        X_lr, X_hr,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=0.1,
        callbacks=[ckpt],
        verbose=1
    )
    if not os.path.isfile(MODEL_PATH):
        model.save(MODEL_PATH)
    print(f"ModelSaved: {MODEL_PATH}", flush=True)
    print("TRAIN DONE", flush=True)

def infer(model_path: str, input_path: str, output_path: str):
    print("INFER START", flush=True)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Haven't found a model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    lr = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if lr is None:
        raise FileNotFoundError(f"Haven't found image: {input_path}")
    lr = cv2.resize(lr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
    sr = model.predict(lr[None, ..., None], verbose=0)[0, ..., 0]
    sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, sr)
    print(f"Saved {output_path}", flush=True)
    print("INFER DONE", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Start education")
    parser.add_argument("--infer", nargs=2, metavar=("INPUT", "OUTPUT"), help="Apply da model")
    args = parser.parse_args()

    if args.infer:
        infer(MODEL_PATH, args.infer[0], args.infer[1])
    else:
        train()
