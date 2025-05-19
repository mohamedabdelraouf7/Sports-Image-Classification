# 🏋️ Sports Image Classification Using CNN

This project is a deep learning application developed as part of the **CSE381 - Introduction to Machine Learning** course at Future University in Egypt. The goal is to classify images of various sports activities using Convolutional Neural Networks (CNNs).

## 👥 Team Members

Alaa Yasser Fathy Bekhit & Mohamed Ahmed Abdelraouf 

**Supervised by:**  
Prof. Mahmoud Ibrahim Khalil  
Eng. Mahmoud Mohamed Soheil

---

## 🎯 Problem Statement

The dataset consists of sports images categorized into different classes (e.g., Badminton, Cricket). The task is to build a multi-class image classifier using CNNs that generalizes well across unseen test data.

- **Training Set**: 8,227 labeled images  
- **Test Set**: 2,056 unlabeled images (for prediction)  
- **Task**: Multi-class classification with robustness to image quality, lighting, and perspective variations.

---

## 📁 Project Structure

- `sports-image-classification-part1.ipynb`: Data loading and preprocessing  
- `sports-image-classification-part2.ipynb`: Augmentation and dataset splitting  
- `sports-image-classification-part3.ipynb`: Model architecture, training, experiments  
- `Sports Image Classification.pdf`: Full project report  
- `README.md`: Project overview and instructions

---

## 🛠️ Preprocessing & Dataset Engineering

- **Duplicate Removal**: Identified using MD5 hashing  
- **Image Resizing**: Standardized to 224×224  
- **Pixel Intensity Analysis**: Grayscale histograms used to evaluate brightness/contrast  
- **Augmentation Techniques**:
  - Rotation, translation, scaling, shearing  
  - Color jittering, Gaussian noise  
- **Dataset Expansion**: Augmented each image, effectively doubling dataset size

---

## 🧪 Data Splitting

Performed using stratified sampling to preserve class distribution:

| Set        | Size   | Purpose               |
|------------|--------|-----------------------|
| Training   | 13,160 | Model learning        |
| Validation | 1,645  | Hyperparameter tuning |
| Test       | 1,645  | Final evaluation      |

---

## 🧠 Model Architecture

- **CNN Backbone**: Custom modular blocks  
- **Squeeze-and-Excitation Block**: Channel-wise attention mechanism  
- **Activation**: Leaky ReLU  
- **Regularization**: L2 weight decay  
- **Framework**: TensorFlow + Keras  

```python
model, history = train_model(
  train_df=train_df,
  val_df=val_df,
  final_img_dir=FINAL_IMG_DIR,
  dropout_rate=0.5,
  weight_decay=0.001,
  optimizer_name='adam',
  lr_scheduler=True,
  initial_lr=1e-4,
  num_conv_blocks=4,
  batch_size=16,
  epochs=30
)
