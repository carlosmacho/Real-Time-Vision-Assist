import threading

import cv2
import torch
import supervision as sv
from ultralytics import YOLO

from src.dto.prediction import Prediction
from src.utils.get_position import get_position
from src.utils.prediction_diff import prediction_add_diff
from src.utils.say_predictions import say_predictions

voice_lock = threading.Lock()
prev_predictions = set()

# Load the YOLOv10 model
model = YOLO('yolov8n.pt')

# Check CUDA availability and move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using {device} for inference")

# Initialize bounding box and label annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to the model's input size
    FRAME_SIZE = (640, 640)
    frame_resized = cv2.resize(frame, FRAME_SIZE)

    left_limit = FRAME_SIZE[0] / 3
    center_limit = left_limit * 2

    # Convert frame to RGB and normalize it
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0  # Normalize to [0, 1]

    # Transpose and add batch dimension
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    # Move frame to the correct device
    frame_tensor = frame_tensor.to(device)

    # Perform object detection using YOLOv10 with NMS and IoU settings
    with torch.no_grad():  # Disable gradient calculation for inference
        results = model.predict(frame_tensor, imgsz=640, conf=0.5, iou=0.3, agnostic_nms=False)

    # Extract predictions from the model output
    if isinstance(results, list):
        predictions = results[0]
    elif isinstance(results, dict):
        predictions = results.get('pred', None)  # Adjust if the key differs

    if predictions is None:
        print("No predictions available")
        continue

    # Convert predictions to supervisely format
    detections = sv.Detections.from_ultralytics(predictions)  # .with_nms(threshold=0.7, class_agnostic=False)

    curr_predictions = set()
    for i in range(len(detections)):
        class_id = detections.data['class_name'][i]
        position_xy = detections.xyxy[i]
        position = get_position(position_xy[0], position_xy[2], left_limit, center_limit)

        curr_predictions.add(Prediction(class_id, position))

    # store sentances already said
    predict_diff = prediction_add_diff(curr_predictions, list(prev_predictions))
    if len(predict_diff) > 0 and not voice_lock.locked():
        print("Predict diff", predict_diff)
        curr_voice_thread = threading.Thread(target=say_predictions,
                                             args=(voice_lock, [str(prediction) for prediction in predict_diff],))
        curr_voice_thread.start()

    print(curr_predictions, prev_predictions)
    prev_predictions = curr_predictions

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
