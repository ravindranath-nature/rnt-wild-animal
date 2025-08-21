# An End-to-End DL-based wildlife detection pipeline with integrated chatbot and deployment tools

An End-to-End UAV-based wildlife detection dashboard using AI, DL, LLM, and deployment tools. This framework have full capability of object detection starting from data annotation, augmentation, model training, testing, and deployment into desktop and edge devices.
It also features an interactive dashboard for real-time monitoring and an LLM-powered assistant for natural language querying, showcasing full-stack ML engineering from research to deployment.

## Goals

The goal of this project is to create a deep learning framework for the wildlife where annotation, augmentation, training, and deployment can happen within a single platform.

## Features

### DataSet Studio

- Build quality datasets from scratch
- Chat Support powered by large language models
  ![Gif of the dataset studio](/screenshots/add_class.gif)

- Annotate in record time
  ![Gif of the assist feature](/screenshots/assist.gif)

- Splitting dataset made easy
  ![Gif of the assist feature](/screenshots/split.gif)

- Augmentation preview and export
  ![Gif of the augmentation feature](/screenshots/augmentation.gif)

### Training Lab

- A simple UI to setup hyper parameters & monitor training
  ![Gif of the training feature](/screenshots/trainingloop.gif)

### One click deployments

- Quickly deploy models to ONNX and Tensorflow and test in realtime.
  ![Gif of the augmentation feature](/screenshots/deployment.gif)

## Data

The data for this project is a collection of images of wild animals. The images are labeled with the type of animal and the position of the animal in the image. The images are stored in the `data` directory.

## Models

The models for this project are stored in the `models` directory. The models are trained using the data in the `data` directory.

## Getting Started

- Clone this repository

```bash
git clone https://github.com/ravindranath-nature/rnt-wild-animal.git
```

- Move into the project directory

```bash
cd rnt-wild-animal
```

- Open your favourite IDE like [VSCode](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/)

```bash
cd code .
```

- Create a virtual environment
  - Windows

```bash
python -m venv .env
```

- - MacOS

```bash
python3 -m venv .env
```

- Annotate your data using a tool like [CVAT](https://github.com/cvat-ai/cvat)
- Move to a model dir that you'd like to train.
- To train say, yolov12,

```bash
cd ./models/yolov12
```

- Update the dataset.yaml file as per your classes
- To begin training

```bash
python train.py
```
