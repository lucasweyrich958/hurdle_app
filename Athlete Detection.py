import torch
import cv2

# Load custom-trained model for hurdles and finish lines
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/Lucas/yolov5/runs/train/yolov5_hurdles/weights/best.pt')

# Load pre-trained model for athletes (humans)
athlete_model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Function to detect objects in a frame using the appropriate model
def detect_objects(frame):
    # Run inference with the model for detecting hurdles and finish lines
    results_hurdles = model(frame)
    detections_hurdles = results_hurdles.xyxy[0]  # Extract bounding boxes and other results

    # Run inference with the athlete_model for detecting humans (athletes)
    results_athletes = athlete_model(frame)
    detections_athletes = results_athletes.xyxy[0]  # Extract bounding boxes and other results

    return detections_hurdles, detections_athletes

# Function to process a frame and draw bounding boxes
def process_frame(frame, detections_hurdles, detections_athletes):
    # Process hurdles and finish lines detections
    for *box, conf, cls in detections_hurdles:
        if cls == 0:
            label = f'Hurdle {conf:.2f}'
            color = (255, 0, 0)  # Red color for hurdles
        elif cls == 1:
            label = f'Finish Line {conf:.2f}'
            color = (0, 255, 255)  # Yellow color for finish line
        else:
            continue  # Skip other classes for this model

        plot_box(frame, box, label, color)

    # Process athletes detections
    for *box, conf, cls in detections_athletes:
        if cls == 0:  # Assuming class 0 is for humans in athlete_model
            label = f'Athlete {conf:.2f}'
            color = (0, 255, 0)  # Green color for athletes
            plot_box(frame, box, label, color)

    return frame

# Function to plot a bounding box with label
def plot_box(frame, box, label, color):
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))
    cv2.rectangle(frame, pt1, pt2, color, 2)
    cv2.putText(frame, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Process video frames
cap = cv2.VideoCapture('/Users/Lucas/Documents/Hurdle_App/Races/iaaf 2018 f.mp4')

# Get the original frame rate of the video
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(original_fps / 5)  # Number of frames to skip to achieve 5 fps

while cap.isOpened():
    for _ in range(frame_interval - 1):
        ret = cap.grab()
        if not ret:
            break

    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    detections_hurdles, detections_athletes = detect_objects(frame)

    # Process the frame with detected objects
    frame = process_frame(frame, detections_hurdles, detections_athletes)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
