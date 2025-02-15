## 🔍Project Overview

### 📌 Title
- Deep Learning Based Skin Cancer Classification

### 📌 Objective
- This project aims to accurately classify skin cancer images into one of seven types using an AI model. This classification system is designed to enhance the reliability of doctors' diagnoses and improve the efficiency of skin cancer treatment.

### 📌 Team Member
- 이승헌, 이주원, 최승우, 최유종

### 📌 Duration
- 2024.04 ~ 2024.06

## 🔍Dataset
### 📌 Source
- We used the **Skin Cancer MNIST: HAM10000 dataset** for our project. This dataset contains images of seven different types of skin cancer, with a total of 10,015 images. This dataset is an open dataset provided by the ISIC archive.
- Each image has a resolution of 600x450 pixels, and the dataset includes not only images but also metadata associated with each image. The metadata contains various details such as the patient's age, gender, and lesion location. 

### 📌 Distribution

- As you can see in the graph, label 0 is overwhelmingly dominant. Considering this data distribution, the most critical challenge in training the model was how to handle the imbalance in the dataset.

## 🔍Project Flow
### 1. Data Splitting
- Out of a total of 10,015 data samples, we divided them into training(6410), validation(1602), and test(2003) sets.

