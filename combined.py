
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import qai_hub as hub
import torch
import cv2
from typing import Tuple, List

# Define the COCO class names
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

# Resize the image to the required dimensions (2048x1024)
resized_image = original_image.resize((2048, 1024))

# Convert the resized image to a numpy array and normalize pixel values to [0, 1]
resized_img_array = np.array(resized_image, dtype=np.float32) / 255.0

# Add a batch dimension to the image array to match model input requirements
input_array = np.expand_dims(resized_img_array, axis=0)

# Define the device and model path
device = hub.Device("QCS6490 (Proxy)")
model = "ffnet_40s_quantized.tflite"

# Submit a profiling job to Qualcomm AI Hub, specifying the model and device
profile_job = hub.submit_profile_job(
    model=model,
    device=device,
    options="--compute_unit npu"
)

# Submit an inference job using the FFNet-40S-Quantized model
inference_job = hub.submit_inference_job(
    model=model,
    device=device,
    inputs=dict(image=[input_array]),
)
on_device_output = inference_job.download_output_data()

# Post-process the output to create a segmented image
output_name = list(on_device_output.keys())[0]
segmentation_output = on_device_output[output_name][0]

# Convert the output to a segmented image
segmentation_map = np.argmax(segmentation_output, axis=-1)

# Remove any singleton dimensions
segmentation_map = np.squeeze(segmentation_map)

# Define a function to map label to color
def label_to_color_image(label):
    """Assigns colors to the label."""
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")
    return cityscapes_colormap[label]

segmentation_colored = label_to_color_image(segmentation_map)

# Convert segmentation_colored to Image and resize to match the original image
segmentation_colored_img = Image.fromarray((segmentation_colored).astype(np.uint8)).resize((original_width, original_height))

# Overlay the segmentation on the original image using PIL
overlay = Image.blend(original_image, segmentation_colored_img, alpha=0.5)

# Display the segmentation output
plt.figure(figsize=(10, 5))
plt.imshow(segmentation_colored_img)
plt.axis('off')
plt.title('Segmentation Output')
plt.show()

# Display the original image with segmentation overlay
plt.figure(figsize=(10, 5))
plt.imshow(overlay)
plt.axis('off')
plt.title('Original Image with Segmentation Overlay')
plt.show()
