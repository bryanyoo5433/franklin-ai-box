import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import qai_hub as hub
import torch
import cv2
from typing import Tuple, List
from qai_hub_models.utils.bounding_box_processing import batched_nms

# Define the COCO class names and Cityscapes colormap
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

cityscapes_colormap = np.array([
    [0, 0, 0], [111, 74, 0], [81, 0, 81], [128, 64, 128], [244, 35, 232], 
    [250, 170, 160], [230, 150, 140], [70, 70, 70], [102, 102, 156], 
    [190, 153, 153], [180, 165, 180], [150, 100, 100], [150, 120, 90], 
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
    [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], 
    [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [0, 0, 110], 
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
])

# Define the path to the input image
image_path = "street_scene.jpg"  # Update this path

# Open the image and convert it to RGB format
original_image = Image.open(image_path).convert('RGB')
original_width, original_height = original_image.size

# Display the original image
plt.imshow(original_image)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Resize the image to the required dimensions (2048x1024) for segmentation
segmentation_size = (2048, 1024)
resized_segmentation_image = original_image.resize(segmentation_size)

# Convert the resized image to a numpy array and normalize pixel values to [0, 1]
resized_segmentation_img_array = np.array(resized_segmentation_image, dtype=np.float32) / 255.0

# Add a batch dimension to the image array to match model input requirements
segmentation_input_array = np.expand_dims(resized_segmentation_img_array, axis=0)

# Define the device and models for segmentation and object detection
device = hub.Device("QCS6490 (Proxy)")
segmentation_model = "ffnet_40s_quantized.tflite"
detection_model = "yolov8_det_quantized.tflite"

# Submit a profiling job to Qualcomm AI Hub, specifying the model and device
profile_job = hub.submit_profile_job(
    model=segmentation_model,
    device=device,
    options="--compute_unit npu"
)

# Submit an inference job using the FFNet-40S-Quantized model for segmentation
segmentation_inference_job = hub.submit_inference_job(
    model=segmentation_model,
    device=device,
    inputs=dict(image=[segmentation_input_array]),
)
segmentation_output = segmentation_inference_job.download_output_data()

# Post-process the output to create a segmented image
segmentation_output_name = list(segmentation_output.keys())[0]
segmentation_output_data = segmentation_output[segmentation_output_name][0]
segmentation_map = np.argmax(segmentation_output_data, axis=-1)
segmentation_map = np.squeeze(segmentation_map)

segmentation_colored = cityscapes_colormap[segmentation_map]
segmentation_colored_img = Image.fromarray((segmentation_colored).astype(np.uint8)).resize((original_width, original_height))

# Resize the image to the required dimensions (640x640) for object detection
detection_size = (640, 640)
resized_detection_image = original_image.resize(detection_size)

# Convert the resized image to a numpy array and normalize pixel values to [0, 1]
resized_detection_img_array = np.array(resized_detection_image, dtype=np.float32) / 255.0

# Add a batch dimension to the image array to match model input requirements
detection_input_array = np.expand_dims(resized_detection_img_array, axis=0)

# Submit an inference job using the YOLOv8 model for object detection
detection_inference_job = hub.submit_inference_job(
    model=detection_model,
    device=device,
    inputs=dict(image=[detection_input_array]),
)
detection_output = detection_inference_job.download_output_data()

# Extract the outputs for object detection
detection_output_names = list(detection_output.keys())
detection_output_values = list(detection_output.values())
boxes = torch.tensor(detection_output_values[0][0])
scores = torch.tensor(detection_output_values[1][0])
class_idx = torch.tensor(detection_output_values[2][0])

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

# Rescale the bounding boxes to the original image size
scale_x = original_width / detection_size[0]
scale_y = original_height / detection_size[1]

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

# Overlay the segmentation on the original image using PIL
overlay = Image.blend(Image.fromarray(image_with_boxes_rgb), segmentation_colored_img, alpha=0.5)

# Display the segmentation output
plt.figure(figsize=(10, 5))
plt.imshow(segmentation_colored_img)
plt.axis('off')
plt.title('Segmentation Output')
plt.show()

# Display the original image with object detection and segmentation overlay
plt.figure(figsize=(10, 5))
plt.imshow(overlay)
plt.axis('off')
plt.title('Original Image with Object Detection and Segmentation Overlay')
plt.show()

print("Detected objects:", ", ".join([coco_class_names[int(label)] for label in pred_class_idx[0]]))
