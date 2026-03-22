import torch
from ultralytics import YOLO
from pathlib import Path
import shutil

if __name__ == '__main__':
    ##### CONFIGURATION #####

    # Detect device (CUDA if available, else CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    DATA_YAML = Path("data/detector2_dataset/data.yaml")
    MODEL_OUTPUT_DIR = Path("models")
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    # parameters
    EPOCHS = 100
    BATCH_SIZE = 8
    IMAGE_SIZE = 1280
    MODEL_SIZE = 's'  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)

    # prints
    print(f"Device: {DEVICE.upper()}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Model size: YOLO11{MODEL_SIZE}")

    ##### LOAD YOLO11 MODEL #####

    model = YOLO(f'yolo11{MODEL_SIZE}.pt')   # model-size can be configured above
    print("Loaded YOLO11 model")

    ##### TRAINING #####

    print("Start Training...")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        rect=True,
        mosaic=0.0,
        mixup=0.0,
        project='models/detector2_training',
        name='yolo_detector',
        device=DEVICE,
        workers=8, #only with gpu, otherwise uncomment or set to 0
        save=True,
        plots=True, # reason for all the plots of training metrics in models/detector_training/yolo_detector/
        verbose=True
    )

    print("Training complete")

    ##### SAVE BEST MODEL #####

    print("Saving best model...")

    best_model_path = Path("models/detector2_training/yolo_detector/weights/best.pt")
    output_model_path = MODEL_OUTPUT_DIR / "detector.pt"

    if best_model_path.exists():
        shutil.copy(best_model_path, output_model_path)
        print(f"Model saved to: {output_model_path}")
    else:
        print(f"ERROR: Best model not found at {best_model_path}")

    ##### EVALUATE MODEL #####

    print("Evaluating model on validation set...")

    metrics = model.val()

    print("Validation Metrics:")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"F1-Score:  {(2 * metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.4f}")


    ##### TRAINING COMPLETE #####

    print("##### TRAINING COMPLETE #####")
    print(f"Trained model: {output_model_path}")
    print(f"Training logs: models/detector2_training/yolo_detector/")
