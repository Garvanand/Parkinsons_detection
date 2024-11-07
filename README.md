# Parkinson's Detection Model

This project involves a machine learning model that predicts whether a person has Parkinson's disease (PD) based on their voice samples. The model works by analyzing features extracted from audio recordings (WAV or MP3 format) and classifying them as either healthy or Parkinson's based on voice characteristics.

I tested several machine learning algorithms, including **K-Nearest Neighbors (KNN)**, **Random Forest**, **Linear Regression**, and **Decision Trees**. After comparing their performance using accuracy, F1-score, and precision, I chose **Decision Trees** as the best-performing model.

## Dataset

- **Source**: The dataset used in this project is the **Oxford Parkinson's Disease Detection Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/174/parkinsons).
- **Description**: This dataset contains voice recordings from 31 individuals: 23 with Parkinson's disease (PD) and 8 healthy controls. In total, there are 195 recordings, with approximately 6 recordings per individual. Each recording is represented by a set of voice features such as jitter, pitch, and other characteristics that could indicate PD. The dataset includes a "status" column where `1` represents PD and `0` represents healthy individuals.
- **Data Format**: The data is in CSV format, with each row representing one voice recording. The first column is the patient's name, and the "status" column indicates whether the individual has PD (1) or is healthy (0).

## Dataset Features

The dataset includes the following features:

![image](https://github.com/user-attachments/assets/8cdb90fa-2a55-4ea2-b134-066ac5816c55)


If you have any questions or need further information, you can contact Max Little at **littlem '@' robots.ox.ac.uk**.

For more details, please refer to the following paper:  
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease," *IEEE Transactions on Biomedical Engineering*.

## Approach

1. **Data Preprocessing**:
    - The voice recordings were preprocessed to extract features such as jitter, pitch, and other voice quality metrics.
    - The dataset was split into training and testing sets for model evaluation.

2. **Testing Algorithms**:
    - Many machine learning algorithms were tested:
        - **K-Nearest Neighbors (KNN)**
        - **Random Forest**
        - **Logistic Regression**
        - **Decision Trees**
        - **Naive Bayes**
        - **And Many More**
    - After evaluating the models based on F1-score, accuracy, and precision, the **Decision Tree** classifier showed the best results, so it was chosen for the final model.

3. **Model Development**:
    - The **Decision Tree** classifier was trained on the dataset, with hyperparameter tuning to improve performance.
    - Cross-validation was used to ensure the model performed well on unseen data.


