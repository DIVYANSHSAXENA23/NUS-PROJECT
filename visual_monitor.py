import argparse
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
from PIL import Image

try:
    # Host-side MobileNetV2 (TensorFlow/Keras) for visual anomaly detection.
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ----------------------------------------------------------------------
# VISUAL ANOMALY MONITOR (ESP32-CAM + MobileNetV2, SEPARATE DEVICE)
# ----------------------------------------------------------------------
# This script runs independently from the micro-seismic real_time_monitor.py.
# It ONLY handles the visual CNN confirmation using frames from an ESP32-CAM.
#
# Model assumptions:
#   - MobileNetV2 transfer-learning model with binary output:
#         0 -> Normal scene
#         1 -> Visual anomaly / obstruction / scene change
#   - Saved as "mobilenet_v2_anomaly.h5" in the same folder.
# ----------------------------------------------------------------------

CNN_MODEL_PATH = "mobilenet_v2_anomaly.h5"
MAX_CNN_RISK = 20  # Visual layer contributes 0–20 risk points

_CNN_MODEL = None  # Lazy-loaded MobileNetV2 model


def load_cnn_model(model_path: str = CNN_MODEL_PATH):
    """Lazy-load the MobileNetV2-based anomaly classifier."""
    global _CNN_MODEL

    if _CNN_MODEL is not None:
        return _CNN_MODEL

    if not _TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow/Keras is not available on this host. "
            "Install TensorFlow to run the visual CNN monitor."
        )

    _CNN_MODEL = load_model(model_path)
    return _CNN_MODEL


def fetch_esp32_cam_image(url: str, timeout: float = 3.0) -> Image.Image:
    """
    Fetch a single JPEG frame from ESP32-CAM over HTTP.

    Example endpoint:
        http://<ESP32_CAM_IP>/capture
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img


def cnn_anomaly_probability(image: Image.Image, model) -> float:
    """
    Run MobileNetV2-based CNN on a single RGB frame and return anomaly probability.
    """
    # MobileNetV2 default input size is 224x224
    img_resized = image.resize((224, 224))
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    p = float(preds.flatten()[0])
    p = max(0.0, min(1.0, p))
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Visual anomaly monitor using ESP32-CAM + MobileNetV2"
    )
    parser.add_argument(
        "--esp32-cam-url",
        type=str,
        required=True,
        help="ESP32-CAM HTTP capture endpoint (e.g. http://<ip>/capture)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between frames when running continuously.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture and evaluate a single frame, then exit.",
    )
    args = parser.parse_args()

    try:
        model = load_cnn_model()
    except Exception as exc:
        print(f"[ERROR] Could not load CNN model: {exc}")
        return

    print("Visual anomaly monitor started.")
    print(f"ESP32-CAM endpoint: {args.esp32_cam_url}")
    print(f"Model path        : {CNN_MODEL_PATH}")
    print(f"Max visual risk   : {MAX_CNN_RISK}")
    print("=" * 70)

    try:
        while True:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                frame = fetch_esp32_cam_image(args.esp32_cam_url)
                p_anomaly = cnn_anomaly_probability(frame, model)
                cnn_risk = int(MAX_CNN_RISK * p_anomaly)

                print(f"Time: {timestamp_str}")
                print(f"  Visual anomaly probability: {p_anomaly:.2f}")
                print(f"  Visual risk contribution  : {cnn_risk}/{MAX_CNN_RISK}")
                print("-" * 70)
            except Exception as exc:
                print(f"[ERROR {timestamp_str}] Failed to process frame: {exc}")

            if args.once:
                break

            time.sleep(max(0.1, args.interval))

    except KeyboardInterrupt:
        print("\nStopping visual monitor.")

from PIL import Image

dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
p = cnn_anomaly_probability(dummy, load_cnn_model())
print("Dummy image anomaly probability:", p)


    
if __name__ == "__main__":
    main()