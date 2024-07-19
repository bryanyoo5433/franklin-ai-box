# Franklin AI Box User Manual

Welcome to the Franklin AI Box User Manual. This guide will help you set up and run the AI Box using the YOLOv8 model on a Qualcomm NPU.

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

## Example Code

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
