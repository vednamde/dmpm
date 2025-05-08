import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load dataset
file_path = "titanic.csv"
df = pd.read_csv(file_path)
print("Dataset loaded.")

# 2. Normalize column names
df.columns = df.columns.str.lower().str.strip()

# 3. Select relevant columns
required_cols = ['survived', 'sex', 'pclass', 'age', 'fare']
df = df[required_cols]

# 4. Drop missing values
df.dropna(inplace=True)

# 5. Encode categorical variables
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # male=1, female=0

# 6. Features and target
X = df[['sex', 'pclass', 'age', 'fare']]
y = df['survived']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 9. Predictions and Evaluation
y_pred = dt_model.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision : {precision_score(y_test, y_pred):.2f}")
print(f"Recall    : {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Plot the Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree")
plt.show()

# 11. Cross-Validation (V-Fold)
cv_scores = cross_val_score(dt_model, X, y, cv=5)
print(f"\n5-Fold Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

# 12. Hyperparameter Optimization using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("\nBest Hyperparameters:")
print(grid_search.best_params_)
print(f"Best CV Accuracy: {grid_search.best_score_:.2f}")