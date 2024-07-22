from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List

coco_class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Load and preprocess the image
image_path = "testimage.jpg"
image = Image.open(image_path).convert('RGB').resize((640, 640))
image_np = np.array(image, dtype=np.float32)
img_array = image_np / 255.0
input_array = np.expand_dims(img_array, axis=0)
print(input_array.shape)


# Submit an inference job using the compiled YOLOv8 model
inference_job = hub.submit_inference_job(
    model=model,
    device=device,
    inputs=dict(image=[input_array]),
)
on_device_output = inference_job.download_output_data()
print("OUTPUT: ", on_device_output)

# Extract the outputs
output_names = list(on_device_output.keys())
output_values = list(on_device_output.values())
print("OUTPUT NAME:", output_names)
print("OUTPUT VALS:", output_values)

# Convert to tensor
boxes = torch.tensor(output_values[0][0])
scores = torch.tensor(output_values[1][0])
class_idx = torch.tensor(output_values[2][0])

# Process the results as needed
print("Boxes:", boxes)
print("Scores:", scores)
print("Class Indices:", class_idx)

# Apply Non-Maximum Suppression (NMS)
nms_iou_threshold = 0.7
nms_score_threshold = 0.45
processed_boxes, processed_scores, processed_class_idx = batched_nms(
    nms_iou_threshold,
    nms_score_threshold,
    boxes,
    scores,
    class_idx,
)

# Convert the predictions to numpy arrays
pred_boxes = [box.cpu().numpy() for box in processed_boxes]
pred_scores = [score.cpu().numpy() for score in processed_scores]
pred_class_idx = [cls_idx.cpu().numpy() for cls_idx in processed_class_idx]

# Collect detected labels
detected_labels = [coco_class_names[int(label)] for label in pred_class_idx[0]]

# Draw the bounding boxes and labels on the image
def draw_boxes(image, boxes, labels, color=(0, 0, 255), size=2):
    img_np = np.array(image)
    for box, label in zip(boxes[0], labels[0]):
        x_min, y_min, x_max, y_max = map(int, box)
        label_name = coco_class_names[int(label)]
        img_np = cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color, size)
        img_np = cv2.putText(img_np, label_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, size)
    return img_np

# Convert the image from RGB to BGR for OpenCV drawing
image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Draw the bounding boxes and labels on the image
image_with_boxes = draw_boxes(image_bgr, pred_boxes, pred_class_idx)

# Convert BGR to RGB for correct color display
image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB).astype(np.uint8)

# Display the image with bounding boxes and labels using PIL
image_with_boxes_pil = Image.fromarray(image_with_boxes_rgb)
image_with_boxes_pil.show()

# Additionally, display the image using matplotlib
plt.imshow(image_with_boxes_rgb)
plt.axis('off')
plt.show()

print("Detected objects:", ", ".join(detected_labels))
