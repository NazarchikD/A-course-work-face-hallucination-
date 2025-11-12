#Face hallucination
# - HOW TO USE? - After using the makes_lowers.py i would get your images with a bad quality or u can put yours then u start this code
# IMPORTANT NOTЕ: Images would be deleted too so save it if you want to

#Important imports
import os
import glob
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

#Paths
MODEL_PATH = "/Users/Nazar/Downloads/CV2/face_hallucination.h5"
SRC_DIR    = "/Users/Nazar/Downloads/CV2/LoweredFaces"
DST_DIR    = "/Users/Nazar/Downloads/CV2/FinishedFaces"


SAVE_METRICS = True 
KEEP_FILENAMES = True
DELETE_SOURCE = True
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass


def list_images(folder):
    files = []
    for ext in ALLOWED_EXT:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)

def model_input_hw(model):

    ishape = getattr(model, "input_shape", None)
    if ishape is None or len(ishape) < 4:
        # fallback
        return 128, 96  # (H,W) дефолт из твоего примера
    _, H, W, C = ishape
    if H is None or W is None:
        # если динамика — берем твои размеры по умолчанию
        return 128, 96
    return int(H), int(W)

def run_model_on_y(model, bgr, in_hw):

    H_in, W_in = in_hw
    H0, W0 = bgr.shape[:2]

    # Конверт в YCrCb
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = [ycrcb[..., i] for i in range(3)]

    # Подготовка входа для модели
    Y_resized = cv2.resize(Y, (W_in, H_in), interpolation=cv2.INTER_CUBIC)
    Y_in = (Y_resized.astype(np.float32) / 255.0)[None, ..., None]

    # Предсказание
    Y_out = model.predict(Y_in, verbose=0)[0, ..., 0].astype(np.float32)

    # Возвращаемся к масштабу исходника
    Y_out_u8_rescaled = np.clip(Y_out * 255.0, 0, 255).astype(np.uint8)
    Y_out_u8 = cv2.resize(Y_out_u8_rescaled, (W0, H0), interpolation=cv2.INTER_CUBIC)

    # Собираем цвет: используем исходные Cr,Cb (сохраняем цветотон/насыщенность исходника)
    res_ycrcb = np.dstack([Y_out_u8, Cr, Cb])
    res_bgr = cv2.cvtColor(res_ycrcb, cv2.COLOR_YCrCb2BGR)
    return res_bgr, Y_out_u8, Y

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_out_path(dst_dir, src_path):
    name = os.path.basename(src_path) if KEEP_FILENAMES else None
    if not name:
        name = f"restored_{os.path.splitext(os.path.basename(src_path))[0]}.png"
    root, ext = os.path.splitext(name)
 
    return os.path.join(dst_dir, f"{root}.png")


def main():
    ensure_dir(DST_DIR)

    # Грузим модель один раз
    print("Downloading model..")
    model = load_model(MODEL_PATH)

    # Определяем входные размеры модели (H,W)
    H_in, W_in = model_input_hw(model)
    print(f"Model's input: (H,W)=({H_in},{W_in})")

    files = [p for p in list_images(SRC_DIR) if os.path.isfile(p)]
    if not files:
        print("⚠️ В папке нет изображений.")
        return

    ok, fail = 0, 0
    psnr_vals, ssim_vals = [], []

    for src_path in tqdm(files, desc="Обработка"):
        try:
            bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"⚠️ Не удалось прочитать: {os.path.basename(src_path)}")
                fail += 1
                continue

            # Прогон через модель
            res_bgr, Y_out_u8, Y_in_u8 = run_model_on_y(model, bgr, (H_in, W_in))

            # Метрики по Y (к исходному Y)
            if SAVE_METRICS:
                psnr_val = cv2.PSNR(Y_in_u8, Y_out_u8)
                ssim_val = ssim(Y_in_u8, Y_out_u8, data_range=255)
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)

            # Сохранение результата
            out_path = safe_out_path(DST_DIR, src_path)
            cv2.imwrite(out_path, res_bgr)

            # Удаление исходника
            if DELETE_SOURCE:
                os.remove(src_path)

            ok += 1

        except Exception as e:
            print(f"Error [{os.path.basename(src_path)}]: {e}")
            fail += 1

    print(f"\nSuccsesful!: {ok}, Ошибок: {fail}")
    if SAVE_METRICS and psnr_vals:
        print(f"MiddlePSNR(Y): {np.mean(psnr_vals):.2f} dB, "
              f"Middle SSIM(Y): {np.mean(ssim_vals):.4f}")

if __name__ == "__main__":
    main()
