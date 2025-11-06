# 🛑 Traffic Sign Recognition (GTSRB)

A deep learning project that uses a **Convolutional Neural Network (CNN)** to automatically recognize and classify traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.  
The goal of this project is to simulate how self-driving cars can detect and understand road signs in real-time, improving safety and automation on the road.

---

## 🚀 Project Overview

This project trains a lightweight CNN model from scratch to classify **43 different types of German traffic signs**.  
It uses image preprocessing, data augmentation, and callback optimization to achieve extremely high accuracy on unseen data — reaching **99.4% validation accuracy**.

The final model is saved and evaluated using TensorFlow/Keras and visualized using Matplotlib.

---

## 📂 Dataset

- **Dataset Name:** German Traffic Sign Recognition Benchmark (GTSRB)  
- **Source:** [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed)  
- **Classes:** 43 traffic sign categories  
- **Format:** RGB images in folders by class  
- **Structure:**

data/
└── GTSRB/
├── Final_Training/
│ └── Images/
└── Final_Test/
└── Images/


---

## ⚙️ Features & Techniques

### 🧩 Data Pipeline
- Images are **decoded, resized, and normalized** to make them memory-efficient.
- **Augmentation** (random brightness, flipping, contrast) improves generalization.
- Data is processed into TensorFlow datasets for faster GPU training.

### 🧠 Model Architecture (CNN)
- Custom CNN built using TensorFlow/Keras.
- Includes multiple **Conv2D, MaxPooling2D, Dropout, and Dense** layers.
- Dropout prevents overfitting, while ReLU and Softmax handle activation and classification.

### ⏱️ Callbacks (Training Safety Features)
Three key callbacks were used:
1. **ModelCheckpoint** — Saves the best model whenever validation accuracy improves.
2. **EarlyStopping** — Stops training if no progress is made, restoring best weights.
3. **ReduceLROnPlateau** — Reduces learning rate when loss stops improving.

✅ The model trained for **13 epochs** and reached a **best validation accuracy of 99.4%**.

---

## 📊 Results

| Metric | Score |
|--------|--------|
| **Training Accuracy** | 99.8% |
| **Validation Accuracy** | **99.4%** |
| **Validation Loss** | 0.0246 |

---

## 🧾 Visualizations

### Model Performance
- **Accuracy vs Epochs**
- **Loss vs Epochs**

Both plots show steady improvement with minimal overfitting.

### Predictions
Random test images are displayed alongside their **predicted** and **true** labels for visual verification.  
Example:

Pred: 28 | True: 28
Pred: 2 | True: 2
Pred: 9 | True: 9


---

## 🧠 Key Learnings

- Understanding how CNNs process images through multiple feature extraction layers.  
- Importance of **data preprocessing** and **augmentation** in improving model robustness.  
- Role of **callbacks** in preventing overfitting and saving computational time.  
- How to evaluate, visualize, and interpret model performance in TensorFlow.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Visualization:** Matplotlib, Seaborn  
- **Data Handling:** NumPy, Pandas  

---

## 🧩 Project Structure
Traffic_Sign_Recognition/
│
├── data/
│ └── GTSRB/
│ └── Final_Training/
│
├── models/
│ └── best_gtsrb.h5
│
├── notebook/
│ └── traffic_sign_recognition.ipynb
│
├── README.md
└── requirements.txt


### 🧪 How to Run the Project

#### 1. Install dependencies
pip install -r requirements.txt

#### 2. Add the dataset

Download the GTSRB dataset and extract it to:

data/GTSRB/Final_Training/Images/

#### 3. Run the notebook
jupyter notebook notebook/traffic_sign_recognition.ipynb

#### 4. Evaluate the model

The model achieves ~99.4% accuracy on the validation dataset.

## Conclusion

This project demonstrates how computer vision models can effectively recognize and classify real-world traffic signs.
With proper training, these techniques can be applied to autonomous driving systems, driver assistance, or smart traffic management.

## Author

Linet Lydia Kagundu
📍 Nairobi, Kenya
🎓 Data Science Student | Open University of Kenya
💼 LinkedIn
 | GitHub

## References

German Traffic Sign Recognition Benchmark (GTSRB)

TensorFlow & Keras Official Documentation

Deep Learning with Python — François Chollet


