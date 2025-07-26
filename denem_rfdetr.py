# Fix PyTorch compatibility issue for older versions
import cv2
import supervision as sv
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
import time

model = RFDETRNano()

cap = cv2.VideoCapture(0)

# FPS calculation variables
fps_counter = 0
fps_start_time = time.time()
fps = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Start timing for this frame
    frame_start_time = time.time()

    # Convert BGR to RGB properly without negative strides
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    detections = model.predict(frame_rgb, threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    # Calculate FPS
    fps_counter += 1
    if fps_counter % 30 == 0:  # Update FPS every 30 frames
        current_time = time.time()
        fps = 30 / (current_time - fps_start_time)
        fps_start_time = current_time

    # Add FPS text to frame
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add detection count
    cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
