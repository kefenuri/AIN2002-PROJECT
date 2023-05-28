import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load real-world data
real_data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Load synthetic data
synthetic_data = pd.read_csv('train.csv')

# Merge the datasets
merged_data = pd.concat([real_data, synthetic_data])

# Handle missing or invalid entries
cleaned_data = merged_data.dropna()

# Encode categorical features
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoder = OrdinalEncoder()
cleaned_data_copy = cleaned_data.copy()
cleaned_data_copy[categorical_features] = encoder.fit_transform(cleaned_data_copy[categorical_features])

# Split the cleaned data into features and target variable
X_cleaned = cleaned_data_copy.drop('stroke', axis=1)
y_cleaned = cleaned_data_copy['stroke']

# Create a pair plot to visualize relationships between variables
sns.pairplot(cleaned_data_copy, hue='stroke')
plt.show()

# Calculate the correlation matrix
correlation_matrix = cleaned_data_copy.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Scale the numerical features
scaler = StandardScaler()
X_cleaned_scaled = scaler.fit_transform(X_cleaned)

# Split the scaled data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_cleaned_scaled, y_cleaned, test_size=0.2, random_state=42)

# Initialize and train the model with increased max_iter and different solver
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Perform cross-validation on the model
cv_scores = cross_val_score(model, X_cleaned, y_cleaned, cv=5)
mean_cv_score = cv_scores.mean()
print("Mean CV Score:", mean_cv_score)

# Generate classification report
classification_report = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report)

# Generate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)

# Load the test data
test_data = pd.read_csv('test.csv')

# Encode categorical features in the test set
test_data_copy = test_data.copy()
test_data_copy[categorical_features] = encoder.transform(test_data_copy[categorical_features])

# Remove the target variable from the test set
X_test = test_data_copy.drop('stroke', axis=1)  # Replace 'column_name' with the actual column name in the test dataset

# Scale the numerical features in the test set
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test set
test_predictions = model.predict(X_test_scaled)

print("Test Predictions:\n", test_predictions)



