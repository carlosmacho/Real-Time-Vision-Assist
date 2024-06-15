import cv2
import torch
import supervision as sv
from ultralytics import YOLOv10

# Load the YOLOv10 model
model = YOLOv10('yolov8n.pt')

# Check CUDA availability and move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using {device} for inference")

# Initialize bounding box and label annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Desired input size for the YOLO model
input_size = 640  # Adjust as needed to match the model's expected input size

# Parameters for NMS and IoU
conf_threshold = 0.5  # Example confidence threshold
iou_threshold = 0.7   # Example IoU threshold
agnostic_nms = True   # Perform NMS independently of class

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to the model's input size
    frame_resized = cv2.resize(frame, (input_size, input_size))

    # Convert frame to RGB and normalize it
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0  # Normalize to [0, 1]

    # Transpose and add batch dimension
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    # Move frame to the correct device
    frame_tensor = frame_tensor.to(device)

    # Perform object detection using YOLOv10 with NMS and IoU settings
    with torch.no_grad():  # Disable gradient calculation for inference
        results = model.predict(frame_tensor, imgsz=input_size, conf=conf_threshold, iou=iou_threshold, agnostic_nms=agnostic_nms)

    # Extract predictions from the model output
    if isinstance(results, list):
        predictions = results[0]
    elif isinstance(results, dict):
        predictions = results.get('pred', None)  # Adjust if the key differs

    if predictions is None:
        print("No predictions available")
        continue

    # Convert predictions to supervisely format
    detections = sv.Detections.from_ultralytics(predictions)

    # Annotate the frame with bounding boxes and labels
    annotated_image = bounding_box_annotator.annotate(scene=frame_resized, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Display the annotated frame
    cv2.imshow("Webcam", annotated_image)

    # Check for the 'Esc' key press to exit the loop
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
