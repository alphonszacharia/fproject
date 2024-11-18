import cv2
import numpy as np
import os
import uuid
import torch
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# Desired width and height for resizing
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

#loading yolov7 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('yolov7', 'custom', f'./saved_model/best.pt', source='local')  # loading the model


video_path = input("Please enter the path to the video file: ")


cap = cv2.VideoCapture(video_path)  # Initialize video capture from the provided video file


# Check if the video was successfully opened
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Variables for user-defined points
points = []
reference_direction = None
tracked_vehicles = {}  # Track vehicles with unique IDs
violated_vehicles = set()  # Keep track of vehicles that have already violated
movement_threshold = 5  # Threshold to ignore noise in movement

#  folder to save images of violating vehicles
output_folder = "violations"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to capture mouse clicks
def select_points(event, x, y, flags, param):
    global points, reference_direction
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Once two points are clicked, compute the reference direction
        if len(points) == 2:
            # Adjusted y-component to account for inverted y-axis
            reference_vector = (points[1][0] - points[0][0], points[0][1] - points[1][1])
            norm = np.linalg.norm(reference_vector)
            if norm != 0:
                reference_direction = np.array(reference_vector) / norm  # Normalize the reference direction
            print(f"Reference Direction Set: {reference_direction}")

# Set mouse callback function for user input
cv2.namedWindow("Violation Detection")
cv2.setMouseCallback("Violation Detection", select_points)

# Read the first frame to allow the user to define the direction
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Resize frame to desired dimensions
first_frame = cv2.resize(first_frame, (DESIRED_WIDTH, DESIRED_HEIGHT))

# Let the user select two points on the first frame
while True:
    temp_frame = first_frame.copy()
    instruction_text = "Click two points to define allowed direction (A to B)"
    cv2.putText(temp_frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Draw the user-selected points and reference direction
    if len(points) > 0:
        cv2.circle(temp_frame, points[0], 5, (255, 0, 0), -1)
    if len(points) == 2:
        cv2.circle(temp_frame, points[1], 5, (255, 0, 0), -1)
        cv2.arrowedLine(temp_frame, points[0], points[1], (255, 0, 0), 2)
    cv2.imshow('Violation Detection', temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    if len(points) == 2:
        break

# Now start processing the video from the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame

# Define a dictionary to store vehicle centroids and their unique IDs
vehicle_ids = {}
next_vehicle_id = 0

# Process the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to desired dimensions
    frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))

    # Inference using YOLOv5
    results = model(frame)

    # Get the predictions: coordinates, confidence, and class id
    detections = results.xyxy[0].numpy()  # (x1, y1, x2, y2, confidence, class_id)

    # Prepare the final detections after filtering confidence
    final_detections = []
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        if confidence > 0.5 and class_id in [2, 3, 5, 7]:  # Filter for car, motorcycle, bus, truck
            final_detections.append(((x1 + x2) // 2, (y1 + y2) // 2, int(x1), int(y1), int(x2), int(y2), int(class_id)))

    #Assign unique IDs to detected vehicles
    updated_vehicle_ids = {}
    for detection in final_detections:
        center_x, center_y, x1, y1, x2, y2, class_id = detection

        # Find the closest previous centroid
        closest_id = None
        min_distance = float('inf')
        for vehicle_id, prev_center in vehicle_ids.items():
            distance = np.linalg.norm(np.array([center_x, center_y]) - np.array(prev_center))
            if distance < min_distance and distance < 50:  # Threshold to match centroids
                closest_id = vehicle_id
                min_distance = distance

        if closest_id is not None:
            updated_vehicle_ids[closest_id] = (center_x, center_y)
        else:
            # Assign a new ID to this vehicle
            updated_vehicle_ids[next_vehicle_id] = (center_x, center_y)
            next_vehicle_id += 1

    vehicle_ids = updated_vehicle_ids

    # Draw green bounding boxes for all detected vehicles
    for detection in final_detections:
        x1, y1, x2, y2 = detection[2:6]
        vehicle_id = next((id for id, center in vehicle_ids.items() if np.linalg.norm(np.array(center) - np.array([detection[0], detection[1]])) < 10), None)
        if vehicle_id not in violated_vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green bounding box

    # Track vehicles and check for violations
    for vehicle_id, (center_x, center_y) in vehicle_ids.items():
        # Track the vehicle's movement direction
        if vehicle_id in tracked_vehicles:
            prev_center_x, prev_center_y = tracked_vehicles[vehicle_id]
            # Adjusted y-component to account for inverted y-axis
            movement_vector = np.array([center_x - prev_center_x, prev_center_y - center_y])

            # Ignore small movements that could be noise
            if np.linalg.norm(movement_vector) > movement_threshold:
                # Compare the movement vector to the reference direction
                if reference_direction is not None and np.dot(movement_vector, reference_direction) < 0:
                    # Check if vehicle has already been recorded as violating
                    if vehicle_id not in violated_vehicles:
                        # Moving opposite to allowed direction
                        for detection in final_detections:
                            det_center_x, det_center_y, x1, y1, x2, y2, class_id = detection
                            distance = np.linalg.norm(np.array([center_x, center_y]) - np.array([det_center_x, det_center_y]))
                            if distance < 10:  # Threshold for matching centroid to bbox
                                # Draw the expanded red bounding box on the frame for violations
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box

                                # Display violation text with vehicle ID
                                cv2.putText(frame, f'Violation ID: {vehicle_id}', (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                                # Capture and save the image of the violating vehicle
                                vehicle_image = frame[y1:y2, x1:x2]
                                image_filename = os.path.join(output_folder, f"violation_{vehicle_id}_{uuid.uuid4().hex}.png")
                                cv2.imwrite(image_filename, vehicle_image)
                                print(f"Violation detected for Vehicle ID: {vehicle_id}. Image saved as {image_filename}")

                                # Add vehicle ID to the set of violated vehicles
                                violated_vehicles.add(vehicle_id)
                                break
        else:
            # New vehicle detected, initialize tracking
            tracked_vehicles[vehicle_id] = (center_x, center_y)

    # Draw red bounding boxes for all vehicles that have violated
    for vehicle_id in violated_vehicles:
        if vehicle_id in vehicle_ids:
            x1, y1, x2, y2 = next(detection[2:6] for detection in final_detections if np.linalg.norm(np.array([detection[0], detection[1]]) - np.array(vehicle_ids[vehicle_id])) < 10)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red bounding box for violating vehicle

    # Draw the user-selected points and reference direction
    if len(points) == 2:
        cv2.circle(frame, points[0], 5, (255, 0, 0), -1)
        cv2.circle(frame, points[1], 5, (255, 0, 0), -1)
        cv2.arrowedLine(frame, points[0], points[1], (255, 0, 0), 2)

    # Display the total count of violated vehicles at the top of the screen
    total_violations = len(violated_vehicles)
    violation_text = f"Total Violations: {total_violations}"
    cv2.putText(frame, violation_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Violation Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()


