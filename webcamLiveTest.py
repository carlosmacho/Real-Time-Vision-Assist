import cv2
import os
import supervision as sv
from ultralytics import YOLOv10

# Load the YOLOv10 model
model = YOLOv10('best_rb_dataset.pt')

# Initialize bounding box and label annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the desired width and height for the displayed frames
#display_width = 640  # Adjust as needed
#display_height = 480  # Adjust as needed

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to the desired dimensions
    #frame_resized = cv2.resize(frame, (display_width, display_height))

    # Perform object detection using YOLOv10
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate the frame with bounding boxes and labels
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
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
