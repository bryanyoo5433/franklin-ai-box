# AI Box User Manual

Welcome to the AI Box User Manual. This guide will help you set up and run the AI Box using the YOLOv8 model on a Qualcomm NPU.

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
- Qualcomm QCS6490 chipset
- Python 3.8 or higher
- Required libraries: numpy, PIL (Pillow), torch, OpenCV, matplotlib, typing
- Qualcomm AI Hub library (`qai_hub`)

## Installation
1. **Install Python**: Ensure you have Python 3.8 or higher installed on your system.
2. **Install Required Libraries**: Use pip to install the necessary libraries.
   ```bash
   pip install numpy pillow torch opencv-python matplotlib typing
