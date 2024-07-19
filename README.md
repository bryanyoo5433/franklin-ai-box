# Franklin AI Box User Manual

Welcome to the Franklin AI Box User Manual. This guide will help you set up and run the AI Box using the YOLOv8 model on a Qualcomm NPU.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Profiling the Model](#profiling-the-model)
    - [Loading and Preprocessing the Image](#loading-and-preprocessing-the-image)
    - [Running the Inference Job](#running-the-inference-job)
    - [Processing the Results](#processing-the-results)
    - [Drawing Bounding Boxes](#drawing-bounding-boxes)
5. [Example Code](#example-code)
6. [Conclusion](#conclusion)

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
    
### Step 7: Ensure Images and Models are in the Same Directory
Make sure your images and models are in the same directory as your Jupyter notebook for easy access.
Download the YOLOv8-Detection-Quantized Model from this [link](https://aihub.qualcomm.com/iot/models/yolov8_det_quantized?domain=Computer+Vision&useCase=Object+Detection&chipsets=QCS6490).
