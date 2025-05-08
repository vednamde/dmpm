# 🎓 DMPM Lab Viva Q&A Guide

This repository contains well-structured **Viva Questions and Answers** for all the lab assignments from the **Data Mining and Predictive Modeling (DMPM)** course.

These Q&As will help you revise concepts and confidently face your practical viva.

---

## 📘 Table of Contents

1. [Data Exploration & Visualization](#1-data-exploration--visualization)
2. [Linear Regression](#2-linear-regression)
3. [Logistic Regression](#3-logistic-regression)
4. [Decision Tree](#4-decision-tree)
5. [K-Means Clustering](#5-k-means-clustering)
6. [Association Rule Mining](#6-association-rule-mining)

---

## 1. 📈 Data Exploration & Visualization

**Q: What is the purpose of data exploration?**  
A: To understand the dataset, identify patterns, and detect missing values or outliers.

**Q: How did you handle missing values?**  
A: I used `mean` or `median` imputation for numerical columns.

**Q: What is skewness and kurtosis?**  
A: Skewness indicates asymmetry; kurtosis indicates the heaviness of tails in the distribution.

**Q: Why log-transform numerical features?**  
A: To normalize skewed data and make it suitable for statistical analysis.

**Q: How did you split the data?**  
A: Using `train_test_split` with a 70-15-15 ratio for training, validation, and testing.

---

## 2. 📉 Linear Regression

**Q: What does linear regression do?**  
A: Predicts a continuous dependent variable based on independent variables.

**Q: How did you evaluate your model?**  
A: Using **Mean Squared Error (MSE)** and **R² Score**.

**Q: What are residuals?**  
A: The difference between actual and predicted values. Useful for model diagnostics.

**Q: What does an R² Score indicate?**  
A: The proportion of variance in the target explained by the model.

---

## 3. 🧮 Logistic Regression

**Q: When do you use logistic regression?**  
A: For binary classification problems.

**Q: What metrics did you use to evaluate the model?**  
A: Accuracy, Confusion Matrix, Precision, Recall, F1-score, ROC AUC Score.

**Q: What does the ROC curve represent?**  
A: The trade-off between true positive rate and false positive rate at various thresholds.

**Q: How did you define the target variable?**  
A: `mpg_class = 1` if mpg > median, else 0.

---

## 4. 🌳 Decision Tree

**Q: Why use a decision tree?**  
A: It's easy to interpret, doesn’t need feature scaling, and works well for both classification and regression.

**Q: How did you validate the model?**  
A: With accuracy metrics and **5-fold cross-validation**.

**Q: What is GridSearchCV?**  
A: A method to find the best hyperparameters using cross-validation.

---

## 5. 📌 K-Means Clustering

**Q: What is the goal of K-Means?**  
A: To group similar data points into `k` clusters based on distance metrics.

**Q: What is inertia in the Elbow Method?**  
A: The sum of squared distances of samples to their closest cluster center.

**Q: Why use PCA in clustering?**  
A: To reduce dimensions for better visualization.

---

## 6. 🔗 Association Rule Mining

**Q: What is an association rule?**  
A: A rule that implies a relationship between items in a transaction dataset (e.g., {Bread} → {Butter}).

**Q: What do support, confidence, and lift mean?**  
- **Support**: Frequency of itemset in dataset  
- **Confidence**: Probability of consequent given antecedent  
- **Lift**: Strength of rule over random occurrence

**Q: How did you implement it?**  
A: Using `mlxtend.apriori()` for frequent itemsets and `association_rules()` for rules.

**Q: Why sort by lift?**  
A: To prioritize the most meaningful and non-random rules.

---

## 📌 Author

Made with ❤️ by VED  
For academic purposes under the course **Data Mining & Predictive Modeling**

---

## 🛠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- mlxtend

---

