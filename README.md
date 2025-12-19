# Project Structure & File Explanations

This document describes the purpose of each file and directory in the **Palm Leaf Disease Detection** project.

---

## 📂 Root Directory
* **`README.md`**: The main project documentation.
* **`requirements.txt`**: List of Python dependencies (PyTorch, OpenCV, etc.).
* **`.gitignore`**: Specifies which files/folders Git should ignore (e.g., `__pycache__`, large datasets).
* **`main.py`**: The central entry point that connects the model, data, and training loops.

---

## 📁 Data Management
* **`Dataset/`**: Local storage for raw leaf images.
* **`src/data/dataset.py`**: Custom PyTorch Dataset class for loading images and labels.
* **`src/data/transforms.py`**: Data augmentation and preprocessing logic (resizing, normalization).

---

## 📁 Model Architecture
* **`src/models/convnext.py`**: Implementation of the **ConvNeXt** backbone architecture.
* **`src/models/classifier.py`**: The final classification layers tailored for specific disease categories.



---

## 📁 Training & Evaluation
* **`src/training/train.py`**: The core training loop logic.
* **`src/training/validate.py`**: Logic for testing the model on the validation set during training.
* **`src/training/losses.py`**: Custom loss functions (e.g., CrossEntropy).
* **`src/training/scheduler.py`**: Manages learning rate decay over time.
* **`src/evaluation/metrics.py`**: Functions to calculate Accuracy, F1-Score, and Precision.
* **`src/evaluation/evaluate.py`**: Script for final testing on the test dataset.

---

## 📁 Deployment & Utilities
* **`src/api/app.py`**: FastAPI or Flask application to serve the model as a web service.
* **`src/utils/logger.py`**: Handles logging of training progress and errors.
* **`src/utils/checkpoints.py`**: Functions for saving and loading model weights (`.pth`).
* **`configs/config.yaml`**: A configuration file containing all hyperparameters (Learning Rate, Batch Size, etc.).

---

## 📁 Automation & Exploration
* **`scripts/`**: Shell scripts (`.sh`) to automate repetitive tasks like starting a full training run.
* **`notebooks/`**: Jupyter Notebooks used for early data exploration and visualizing results.
* **`checkpoints/`**: Directory where the best-performing model versions are stored.
