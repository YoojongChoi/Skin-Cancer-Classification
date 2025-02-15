## 🔍Project Overview

### 📌 Title
- Deep Learning Based Skin Cancer Classification

### 📌 Objective
- This project aims to accurately classify skin cancer images into one of seven types using an AI model. This classification system is designed to enhance the reliability of doctors' diagnoses and improve the efficiency of skin cancer treatment.
- We conducted a study on skin cancer classification and performance analysis **using data augmentation, CNN, ResNet, and transfer learning with MobileNet, along with hyperparameter optimization.**

### 📌 Team Member
- 이승헌, 이주원, 최승우, 최유종

### 📌 Duration
- 2024.04 ~ 2024.06

## 🔍Dataset
### 📌 Source
- We used the **Skin Cancer MNIST: HAM10000 dataset** for our project. This dataset contains images of seven different types of skin cancer, with a total of 10,015 images. This dataset is an open dataset provided by the ISIC archive.
- Each image has a resolution of 600x450 pixels, and the dataset includes not only images but also metadata associated with each image. The metadata contains various details such as the patient's age, gender, and lesion location. 

### 📌 Distribution
- Labels
![Image](https://github.com/user-attachments/assets/e3e3d9fa-753d-403c-a155-931261f2b4eb)

![Image](https://github.com/user-attachments/assets/4967aa9a-766b-4ee0-bf45-4ed2fec1a7fd)

- As you can see in the graph, label 0 is overwhelmingly dominant. Considering this data distribution, the most critical challenge in training the model was how to **handle the imbalance in the dataset.**

## 🔍Project Flow
### 1. Data Splitting
- Out of a total of 10,015 data samples, we divided them into training(6410), validation(1602), and test(2003) sets.

### 2. Model Design for Performance Analysis
- We designed a total of four models and conducted experiments. The reason for structuring the models in four stages was to compare and analyze how the additional components at each stage affect the model's performance.
  
    1. CNN with Images Only
    2. CNN with Images + Metadata + Residual Connections
    3. Images + Metadata with Data Augmentation
    4. Images + Metadata with Data Augmentation and Transfer Learning

#### 📌 1. CNN with Images Only
- Results
![Image](https://github.com/user-attachments/assets/75bb7a6d-92c4-4926-970e-ea2c4764f099)

![Image](https://github.com/user-attachments/assets/81443e34-9553-487a-ac83-b36ef2ed6357)

 - This model consists of five repeated blocks of convolutional layers, max pooling layers, and dropout layers, followed by a flatten layer and an output layer.
 - In the final epoch, the accuracy on the validation data was approximately 0.76, and the loss was around 0.65. These results indicate that the model can classify images with relatively high accuracy.

#### 📌 2. CNN with Images + Metadata + Residual Connections
- Results
![Image](https://github.com/user-attachments/assets/e8842714-1fec-4b93-81c5-eee539174498)
![Image](https://github.com/user-attachments/assets/e40d4a5e-c4e6-4ca6-8232-12d220f7f8b2)

- This model incorporates residual connections and merges with metadata using a Concatenate layer. It takes both metadata and image data as inputs, combines them, and then produces the final output.
- In the final epoch, the accuracy on the validation data was approximately 0.71, and the loss was around 0.79. Although we expected Model 2 to perform better than Model 1, its performance was lower due to factors such as data imbalance, overfitting issues, and suboptimal hyperparameter settings.

#### 📌 3. Images + Metadata with Data Augmentation
- Data Augmentation

![Image](https://github.com/user-attachments/assets/5ffcc596-449f-4abb-92f0-304947fd2b91)

- Results
![Image](https://github.com/user-attachments/assets/0ef7e052-bd08-4052-80e2-875a2f67af70)
![Image](https://github.com/user-attachments/assets/29036eef-24c4-4e9f-bc4b-5746cee380eb)
![Image](https://github.com/user-attachments/assets/45deac4a-9452-42c0-87dd-fd8f0fc81585)

- This model utilized augmented data and was implemented using CNN. To optimize hyperparameters, we employed random search. Both the validation and training datasets achieved an accuracy of 78.5%.
- Since the training data size nearly doubled compared to the previous models, it can be inferred that the performance has improved.

#### 📌 4. Images + Metadata with Data Augmentation and Transfer Learning
- 4번 결과 사진
![Image](https://github.com/user-attachments/assets/1be8e87e-b7b6-4504-a3f7-46c9c62a08d1)
![Image](https://github.com/user-attachments/assets/8e183747-c841-458d-ae96-366220085b8b)
- We chose **MobileNet** due to its low resource consumption, as our development was conducted on Colab with limited resources.
- The classification was performed using both metadata and extracted image features. The accuracy on the validation and test datasets reached 84%.

## 📝 Evaluation

The lowest performance was observed when using a purely CNN-based approach without additional enhancements. This suggests that training a deep learning model from scratch, **without leveraging external knowledge or additional features, struggles to generalize well with limited data.**

Incorporating metadata into the CNN model, whether in a standard CNN architecture or a ResNet-inspired structure, led to some improvements. However, the performance gains were not as significant as expected. **This highlights the challenge of effectively integrating structured data with image features, especially when the metadata is relatively simple compared to the complexity of the image features.**

The best performance was achieved when using a pre-trained MobileNet model combined with metadata. **This outcome reinforces the power of transfer learning, where labeled data is often scarce.** The fact that a model pre-trained could still be fine-tuned for skin cancer classification suggests deep networks trained on general image datasets can learn transferable features that are useful for domain-specific tasks.

## 📝 Reflection

**One key takeaway from this study is the importance of data augmentation.** Regardless of the model used, applying augmentation consistently led to improved performance. This demonstrates that increasing data diversity helps models generalize better, reducing overfitting and mitigating the impact of data imbalance.

Through this process, I realized how crucial it is to balance model complexity, data quality, and computational efficiency. Initially, I expected that adding metadata would lead to significant improvements, but the results showed that model architecture and feature extraction play a much bigger role. Additionally, optimizing hyperparameters, particularly in deep learning models, proved to be a decisive factor in achieving better performance.

Overall, this study reaffirmed the effectiveness of transfer learning and the necessity of data augmentation in medical image classification. It also highlighted the importance of thoughtful model design and hyperparameter tuning to achieve the best possible results.
