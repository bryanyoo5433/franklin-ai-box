import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import qai_hub as hub  # Hypothetical module for Qualcomm AI Hub interactions
import torch
import cv2
from typing import Tuple, List
from qai_hub_models.utils.bounding_box_processing import batched_nms

# Define the COCO class names
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

# Define the path to the input image
image_path = "street_scene.jpg"

# Open the image and convert it to RGB format
original_image = Image.open(image_path).convert('RGB')
original_width, original_height = original_image.size

# Resize the image to the required dimensions (640x640)
new_size = (640, 640)
resized_image = original_image.resize(new_size)

# Convert the resized image to a numpy array and normalize pixel values to [0, 1]
resized_img_array = np.array(resized_image, dtype=np.float32) / 255.0

# Add a batch dimension to the image array to match model input requirements
input_array = np.expand_dims(resized_img_array, axis=0)

# Define the device and model path for YOLOv8
device = hub.Device("QCS6490 (Proxy)")
model = "yolov8_det_quantized.tflite"

# Submit an inference job using the compiled YOLOv8 model
inference_job = hub.submit_inference_job(
    model=model,
    device=device,
    inputs=dict(image=[input_array]),
)
on_device_output = inference_job.download_output_data()

# Extract the outputs
output_names = list(on_device_output.keys())
output_values = list(on_device_output.values())

# Convert to tensor
boxes = torch.tensor(output_values[0][0])
scores = torch.tensor(output_values[1][0])
class_idx = torch.tensor(output_values[2][0])

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

# Rescale the bounding boxes to the original image size
scale_x = original_width / new_size[0]
scale_y = original_height / new_size[1]

rescaled_boxes = []
for box in pred_boxes[0]:
    x_min, y_min, x_max, y_max = box
    x_min = int(x_min * scale_x)
    y_min = int(y_min * scale_y)
    x_max = int(x_max * scale_x)
    y_max = int(y_max * scale_y)
    rescaled_boxes.append([x_min, y_min, x_max, y_max])

# Draw the bounding boxes and labels on the original image
def draw_boxes(image, boxes, labels, color=(0, 0, 255), size=2):
    img_np = np.array(image)
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        label_name = coco_class_names[int(label)]
        img_np = cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color, size)
        img_np = cv2.putText(img_np, label_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, size)
    return img_np

# Convert the original image from RGB to BGR for OpenCV drawing
original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

# Draw the bounding boxes and labels on the original image
image_with_boxes = draw_boxes(original_image_bgr, rescaled_boxes, pred_class_idx[0])

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


print("Detected objects:", ", ".join(detected_labels))
