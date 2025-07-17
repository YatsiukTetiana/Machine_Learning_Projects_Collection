# 📘 Machine Learning Projects Collection

This repository contains end-to-end machine learning projects focused on regression, classification, and unsupervised learning (clustering). All projects are implemented in Python using libraries like scikit-learn, pandas, matplotlib, and seaborn.

## 🧭 Project Navigation

| Project Name | Description | Link |
|--------------|-------------|------|
| **Linear Regression** | Predicting Revenue Based on Time Spent Online | [📂 Go to notebook](https://colab.research.google.com/drive/1hriUCZZLwE5e1YF-VZr0MLwti7eHrbRc?usp=sharing) [📥 Download .ipynb](Predicting_Revenue_Based_on_Time_Spent_Online.ipynb) |
| **Linear Regression** | Estimation of the Average Customer Check | [📂 Go to notebook](https://colab.research.google.com/drive/1qAt3suHyIndRq_3czRSMJdJvQhshH4dF?usp=sharing) [📥 Download .ipynb](Estimation_of_the_Average_Customer_Check.ipynb) |
| **Logistic Regression** | Credit Scoring | [📂 Go to notebook](https://colab.research.google.com/drive/1WSBYNmVx1N--yre6HKjL-_1W9usq4-fG?usp=sharing) [📥 Download .ipynb](Credit_Scoring.ipynb) |
| **Clustering (K-Means, K-Median, DBSCAN)** | Customer Segmentation | [📂 Go to folder with notebooks](https://drive.google.com/drive/folders/13QBZlCyRFPnsUOYkfieQK9hOvIDim7tQ?usp=sharing) |

---

## Linear Regression  
**Project:** *Predicting Revenue Based on Time Spent Online*  

Built a linear regression model to predict customer purchase value based on time spent on the website. 
Evaluated model performance using MAE, MSE, RMSE, and R² on test data and via cross-validation. 
Analyzed results to determine the most relevant metric for business goals.
👉 [Open notebook](https://colab.research.google.com/drive/1hriUCZZLwE5e1YF-VZr0MLwti7eHrbRc?usp=sharing)
📥 [Download .ipynb](Predicting_Revenue_Based_on_Time_Spent_Online.ipynb)

**Project:** *Estimation of the Average Customer Check*  

Built a multiple linear regression model to predict average customer check using key features like cart size, item price, discount, and time on site. 
Improved the model with Ridge and Lasso regularization, tuning hyperparameters, and evaluating performance with MAE and R². 
Analyzed Lasso coefficients to identify less important features.
👉 [Open notebook](https://colab.research.google.com/drive/1qAt3suHyIndRq_3czRSMJdJvQhshH4dF?usp=sharing)
📥 [Download .ipynb](Estimation_of_the_Average_Customer_Check.ipynb)

## Logistic Regression  
**Project:** *Credit Scoring*

Developed a logistic regression model to predict credit card approval, including data preprocessing, encoding, and feature scaling. Evaluated model performance using accuracy, precision, recall, F1-score, and specificity, highlighting their business relevance. Tuned regularization hyperparameters to improve the model with validation curves.  
👉 [Open notebook](https://colab.research.google.com/drive/1WSBYNmVx1N--yre6HKjL-_1W9usq4-fG?usp=sharing)
📥 [Download .ipynb](Credit_Scoring.ipynb)

## Clustering  

**Project:** *Customer Segmentation with K-means*

Segmented customers based on purchase frequency and total spending using K-means.  
Performed standardization, determined the optimal number of clusters via the elbow method, and visualized results.  
👉 [Open notebook](https://colab.research.google.com/drive/1s_r5ZpXb3HOdSKF87hKMqcAlGVUVvs5W?usp=sharing)
📥 [Download .ipynb](Customer_Segmentation_with_K_means.ipynb)

**Project:** *Customer Segmentation with K-median*

Clustered customers by purchases and spending using the K-median algorithm with Manhattan distance.  
Compared results with K-means, evaluated inertia, and visualized optimal clusters.  
👉 [Open notebook](https://colab.research.google.com/drive/19n2cfNHfwhTVjRwECV7zJI1O-fUC5IO0?usp=sharing)
📥 [Download .ipynb](Customer_Segmentation_with_K_median.ipynb)


**Project:** *Customer Segmentation with DBSCAN*

Segmented users based on browsing time and purchase amount using DBSCAN.  
Tuned `eps` and `min_samples` to identify meaningful clusters and noise points.  
👉 [Open notebook](https://colab.research.google.com/drive/1i4B7cc_h4bAu-7KFcxpxA1UpNyNdd7ft?usp=sharing)
📥 [Download .ipynb](Customer_Segmentation_with_DBSCAN.ipynb)


## 🛠️ Tools & Libraries  
- `Python`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
- Jupyter Notebooks


## Purpose  
These projects demonstrate foundational machine learning techniques applied to real-world-style business data. They highlight skills in:  
- Model development & evaluation  
- Feature engineering  
- Segmentation & insight generation
