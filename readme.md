

# An End-to-End DL-based wildlife detection pipeline with integrated chatbot and deployment tools

This repository is for training different object recognition models to identify wild animals.

## Goals

The goal of this project is to create a model that can identify wild animals in images. The model should be able to identify the type of animal and the position of the animal in the image.

## Data

The data for this project is a collection of images of wild animals. The images are labeled with the type of animal and the position of the animal in the image. The images are stored in the `data` directory.

## Models

The models for this project are stored in the `models` directory. The models are trained using the data in the `data` directory.
![WhatsApp Image 2025-06-07 at 8 21 11 PM](https://github.com/user-attachments/assets/0e7bc080-694f-46ce-89ef-9b8268d8ce62)
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
