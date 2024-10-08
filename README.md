# Franklin AI Box User Manual

Welcome to the Franklin AI Box User Manual. This guide will help you set up and run on-device on the Franklin AI Box. This repository contains code for

1. Object Detection - YOLOv8 Quantized
2. Image Segmentation - FFNet 40s Quantized

<img width="376" alt="Original" src="https://github.com/user-attachments/assets/5f8c1393-3cde-4491-bb7b-333ef6879cda">
<img width="523" alt="Segmentation" src="https://github.com/user-attachments/assets/eb2fefc3-b649-4947-a45d-f55a62f7f47c">
<img width="520" alt="Both" src="https://github.com/user-attachments/assets/d20b215a-3557-4e34-b84f-c627cbab4b32">


## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Example Code](#example-code)
5. [Conclusion](#conclusion)

## Introduction
The AI Box is designed to provide real-time object detection using the YOLOv8 model running on the Qualcomm QCS6490 chipset. This manual will guide you through the steps required to run inference and process the results.

## Requirements
To install the required libraries, you can use the provided `requirements.txt` file. First, ensure you have Python >=3.8 and <=3.10 installed on your system. We recommend using a Python virtual environment (miniconda or virtualenv).

- Qualcomm QCS6490 chipset
- Python 3.8 or higher
- Required libraries: numpy, PIL (Pillow), torch, OpenCV, matplotlib, typing
- Qualcomm AI Hub library (`qai_hub`)

## Installation

### Step 1: Create a Virtual Environment
Creating a virtual environment helps in managing dependencies and avoiding conflicts with other projects.

1. Open a terminal or command prompt.
2. Run the following command to create a virtual environment using `conda`:

   ```bash
   conda create --name myenv python=3.9.7

### Step 2: Activate the Virtual Environment
Activate the virtual environment using the following command:

    ```bash
    conda activate myenv

### Step 3: Install the Required Libraries
Install the necessary libraries using the requirements.txt file.

    ```bash
    pip install -r requirements.txt
    
### Step 4: Install Qualcomm AI Hub Library
Follow the installation instructions on the [Qualcomm AI Hub GitHub page](https://github.com/quic/ai-hub-models).

    ```bash
    pip install qai_hub
    
### Step 5: Set Up Jupyter Lab
Install Jupyter Lab if it's not already installed.

    ```bash
    pip install jupyterlab

### Step 6: Run Jupyter Lab
Start Jupyter Lab from your terminal or command prompt.

    ```bash
    jupyter lab

### Step 7: Organize Your Files
1. Download the Model: Download the YOLOv8-Detection-Quantized Model from this [link](https://aihub.qualcomm.com/iot/models/yolov8_det_quantized?domain=Computer+Vision&useCase=Object+Detection&chipsets=QCS6490).
2. Create a Folder: In Jupyter Lab, create a new folder and place the downloaded model, your Jupyter notebook, and the images you want to use all in this folder.

## Example Code (Object Detection)
1.  If want to do Image Segmentation, go to segmentation.py in the repo.
2.  If want to do Object Detection with Image Segmentation, go to combined.py in the repo.

### Profiling the Model
1. Load and preprocess the image.
2. Submit a profiling job to understand the performance characteristics of the model on the specified device.

   ```python
   import numpy as np
   from PIL import Image
   import matplotlib.pyplot as plt
   import qai_hub as hub  # Hypothetical module for Qualcomm AI Hub interactions
   
   # Define the path to the input image
   image_path = "street_scene.jpg"
   
   # Open the image and convert it to RGB format
   original_image = Image.open(image_path).convert('RGB')
   
   # Resize the image to the required dimensions (640x640)
   new_size = (640, 640)
   image = original_image.resize(new_size)
   
   # Display the resized image
   plt.imshow(image)
   plt.axis('off')
   plt.show()
   
   # Convert the image to a numpy array and normalize pixel values to [0, 1]
   img_array = np.array(image, dtype=np.float32) / 255.0
   
   # Add a batch dimension to the image array to match model input requirements
   input_array = np.expand_dims(img_array, axis=0)
   
   # Define the device and model path for YOLOv8
   device = hub.Device("QCS6490 (Proxy)")
   model = "yolov8_det_quantized.tflite"
   
   # Submit a profiling job to Qualcomm AI Hub, specifying the model and device
   profile_job = hub.submit_profile_job(
       model=model,
       device=device,
       options="--compute_unit npu"
   )
   
   # The profiling job will help to understand the performance characteristics of the model on the specified device
   
   # Note: Ensure to include appropriate error handling and additional functionality as needed
   ```

### Running the Model
1. Preprocess the image.
2. Run the inference job.
3. Postprocess the results.
   
   ```python
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


   ```

## Conclusion
This manual provided a comprehensive guide to setting up and running the YOLOv8 and FFNet model on the Qualcomm NPU of the AI Box. By following these steps, you should be able to profile the model, preprocess images, run inference, and postprocess the results effectively.

