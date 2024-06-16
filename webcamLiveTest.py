import subprocess
import threading
import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

# Load the YOLOv10 model
model = YOLO('C:\Users\Carlos\vsProjects\CP3\runs\detect\train10\weights\best.pt')

# Check CUDA availability and move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using {device} for inference")

# Initialize bounding box and label annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the webcam
cap = cv2.VideoCapture(0)

prev_predictions = set()

LABELS = [
    'person',
]

def prediction_add_diff(curr_predictions, prev_prediction):
    return [curr_predict for curr_predict in curr_predictions if curr_predict not in prev_prediction]


def say_predictions(predict_diff):
    for new_predict in predict_diff:
        subprocess.call(["say", new_predict])

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to the model's input size
    frame_resized = cv2.resize(frame, (640, 640))

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

    # FILTER BY LABELS // UNCOMMENT TO ENABLE FOR SPECIFIC LABELS
    #detections = detections[np.isin(detections.data['class_name'], LABELS,)]

    curr_predictions = set(detections.data['class_name'])
    print(curr_predictions)

    if curr_predictions != prev_predictions:
        predict_diff = prediction_add_diff(curr_predictions, prev_predictions)

        # FILTER BY LABELS // UNCOMMENT TO ENABLE FOR SPECIFIC LABELS
        # predict_diff = [predict for predict in predict_diff if predict in LABELS]

        threading.Thread(target=say_predictions, args=(predict_diff,)).start()

    prev_predictions = curr_predictions
    # if set(detections_class_names) != already_detected:
    #
    #
    # if len(detection_name) > 0 and detection_name[0] not in already_detected:
    #    already_detected.add(detection_name[0])
    #    subprocess.call(["say", detection_name[0]])

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
