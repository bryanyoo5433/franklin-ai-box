import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import qai_hub as hub
import time

# Load and preprocess the image
image_path = "street_scene.jpg"  # Update this path
image = Image.open(image_path).convert('RGB').resize((2048, 1024))  # Convert to RGB to remove alpha channel

# Display the input image
plt.imshow(image)
plt.axis('off')
plt.show()

# Convert the image to numpy array of shape [1, 1024, 2048, 3]
img_array = np.array(image, dtype=np.float32) / 255.0

# Ensure correct layout (NHWC)
input_array = np.expand_dims(img_array, axis=0)

# Define the device and model path
device = hub.Device("QCS6490 (Proxy)")
model = "ffnet_40s_quantized.tflite"


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

# Example color map (adjust according to actual colors)
cityscapes_colormap = np.array([
    [0, 0, 0], [111, 74, 0], [81, 0, 81], [128, 64, 128], [244, 35, 232], 
    [250, 170, 160], [230, 150, 140], [70, 70, 70], [102, 102, 156], 
    [190, 153, 153], [180, 165, 180], [150, 100, 100], [150, 120, 90], 
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
    [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], 
    [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [0, 0, 110], 
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
])

segmentation_colored = label_to_color_image(segmentation_map)

# Convert segmentation_colored to Image and resize to match the original image
segmentation_colored_img = Image.fromarray((segmentation_colored).astype(np.uint8)).resize(image.size)

# Overlay the segmentation on the original image using PIL
overlay = Image.blend(image, segmentation_colored_img, alpha=0.5)

# Display the segmentation output
plt.figure(figsize=(10, 5))
plt.imshow(segmentation_colored_img)
plt.axis('off')
plt.show()

# Display the original image with segmentation overlay
plt.figure(figsize=(10, 5))
plt.imshow(overlay)
plt.axis('off')
plt.show()
