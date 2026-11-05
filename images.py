from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Define paths relative to the script location
BASE_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
INPUT_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "output_images"

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def run_ppe_detection(confidence=0.30):
    """
    Detects PPE using a high-quality, production-grade YOLOv8 model 
    (SafetyVision v2) trained on 80k+ real-world images.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    image_paths = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        print(f"No images found in '{INPUT_DIR}'. Add test images and try again.")
        return

    print("Fetching robust SafetyVision YOLOv8 PPE detection model...")
    
    try:
        # Primary: SafetyVision v2 model (highly robust, 13 classes)
        safe_model_path = hf_hub_download(
            repo_id="ayushgupta7777/safetyvision-yolov8",
            filename="v2/best.pt"
        )
    except Exception as e:
        print(f"Failed to download primary model: {e}")
        print("Fetching fallback verified model...")
        # Fallback: Baskarmother V1 public model (17 classes)
        safe_model_path = hf_hub_download(
            repo_id="baskarmother/yolov8-ppe-construction",
            filename="best.pt"
        )
    
    # Load the model using the secure local path
    model = YOLO(safe_model_path)

    print(f"\nProcessing {len(image_paths)} image(s) from '{INPUT_DIR.name}'...\n")

    for img_path in image_paths:
        # Run inference 
        results = model.predict(source=str(img_path), conf=confidence, verbose=False)
        
        # Extract class names from the detected bounding boxes
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            detected_items = [model.names[int(b.cls[0])] for b in boxes]
            
            # Count occurrences of each detected item for cleaner terminal output
            item_counts = {item: detected_items.count(item) for item in set(detected_items)}
            items_str = ", ".join([f"{count}x {item}" for item, count in item_counts.items()])
        else:
            items_str = "None detected (Try lowering confidence?)"
            
        print(f"Image: {img_path.name}")
        print(f"  Detected Items: {items_str}")
        
        # Save annotated image directly to output_images/
        save_path = OUTPUT_DIR / img_path.name
        results[0].save(filename=str(save_path))
        print(f"  Saved output to: {save_path}\n" + "-" * 40)

if __name__ == "__main__":
    # 0.30 confidence is standard. If the clear safety glasses in the second image 
    # are still being missed, lower this to 0.15.
    run_ppe_detection(confidence=0.30)