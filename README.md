"# Capstone Project" 
FindDefault (Prediction of Credit Card fraud)
Problem Statement:
A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
We have to build a classification model to predict whether a transaction is fraudulent or not.

Purpose of the problem Statement:
The purpose of the problem statement is to address the critical issue of credit card fraud detection, which poses significant risks to both consumers and financial institutions. By building a robust classification model, we aim to accurately identify fraudulent transactions, thereby protecting customers from unauthorized charges and maintaining trust in credit card services.
Key objectives include:
Data Collection: Obtain transaction data from September 2013, focusing on European cardholders.
Exploratory Data Analysis (EDA): Analyse the dataset to assess its quality, address missing values, and identify outliers.
Data Preprocessing: Ensure the correct data type for the date column and perform necessary data transformations.
Balancing Data: Address the highly unbalanced nature of the dataset by applying techniques to balance the class distribution, enhancing the model's ability to detect fraudulent transactions.
Feature Engineering and Selection: Create and select relevant features to improve the model's predictive performance.
Train/Test Split: Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.
Model Evaluation Metrics: Choose appropriate evaluation metrics to assess the model's performance, such as accuracy, precision, recall, and F1-score.
Model Selection and Training: Select a suitable classification algorithm and train the model on the training data.
Hyperparameter Tuning: Fine-tune the model's hyperparameters to optimize its performance, using techniques like grid search or randomized search.
Model Deployment Plan: Develop a plan for deploying the trained model into production, ensuring seamless integration into credit card company systems for real-time fraud detection.
By addressing these tasks and activities, we aim to develop an effective credit card fraud detection system that enhances security and safeguards customers against fraudulent activities, ultimately fostering trust and confidence in credit card services.

Data Source and Methodology:
The data source for the problem statement is a dataset containing transactions made by credit cards in September 2013 by European cardholders. The dataset is provided as a CSV file i.e., creditcard.csv.
Methodology for addressing the problem statement typically involves several steps:
Data Collection
Exploratory Data Analysis (EDA)
Balancing Data
Feature Engineering and Selection
Train/Test Split
Model Evaluation Metrics
Model Selection and Training
Hyperparameter Tuning
Model Deployment Plan
Overall, the methodology involves a systematic approach to data preprocessing, model training, evaluation, and deployment, aimed at developing an effective credit card fraud detection system.


Code and Models:
The code for this project involves several steps, including data preprocessing, model selection, training, evaluation, and validation.
Data Preprocessing: This step involves handling missing values, scaling numerical features, and encoding categorical variables.
Model Selection: Various machine learning algorithms such as logistic regression, random forest, and gradient boosting classifiers can be considered for building the fraud detection model.
Model Training and Evaluation: The selected machine learning model is trained using the preprocessed data and evaluated using appropriate evaluation metrics such as accuracy,confusion matrix, and classification report are used to evaluate the classification model's performance on the test data.The ROC curve and AUC (Area Under the ROC Curve) are computed to assess the model's ability to distinguish between the positive and negative classes across different thresholds.
Model Serialization: Once the model is trained and evaluated, it can be serialized using libraries like pickle or joblib for future use.

How to run the code and reproduce the results:
Dataset - creditcard.csv
Dependencies:
List of libraries and dependencies:
NumPy
pandas
scikit-learn
Matplotlib
Seaborn
Sklearn.model_selection - train_test_split
Imblearn.over_sampling -  SMOTE
Sklearn.ensemble - RandomForestClassifier
Sklearn.metrics - accuracy_score,roc_curve, roc_auc_score,confusion_matrix,classification_report
sklearn.model_selection - GridSearchCV


Data Preparation:
Explain the steps to prepare the data:
Loading the creditcard.csv dataset.
Split the dataset into features (X) and target variable (y).
Split the data into training and testing sets.


Model Training:
Provide instructions on training the classification model:
 Random Forest, Logistic Regression.
Fit the model to the training data.


Model Evaluation:
Detail the process of evaluating the model's performance:
Calculate evaluation metrics such as accuracy. 
Plot the confusion matrix and classification report.
Compute the ROC curve and AUC score.


Reproducing Results:
Provide step-by-step instructions to reproduce the results:
1.Downloading and loading the data i.e., creditcard.csv
2.Importing the necessary libraries 
3.Perform exploratory data analysis (EDA) to understand the distribution of features, identify any missing values or outliers, and explore relationships between variables.
4.Preprocess the data by handling missing values, scaling numerical features, encoding categorical variables, and addressing class imbalance if necessary.
5.Split the dataset into training and testing sets (e.g., using train_test_split from scikit-learn).
6.Choose a machine learning algorithm suitable for classification tasks (e.g., RandomForestClassifier, LogisticRegression, etc.).
7.Train the chosen model on the training data using fit() method.
8.Use the trained model to make predictions on the testing data using predict() method.Evaluate the model's performance using various metrics such as accuracy, accuracy score, confusion matrix, ROC curve, and AUC.
9.perform hyperparameter tuning to optimize the model's performance using techniques like grid search or random search.
10.Serialize the trained model using libraries like pickle or joblib to save it as a file (e.g., creditcard_model.pkl).

Visualize the evaluation results using plots and charts to gain insights into the model's performance.
Load the trained model ( rf = RandomForestClassifier(random_state=42)).
Load the creditcard.csv dataset - data = pd.read_csv("creditcard.csv").
Preprocess the data (if necessary).
Make predictions using the loaded model - 
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
Evaluate the model's performance.Visualize the evaluation results-

Accuracy: 0.9441624365482234
Confusion Matrix:
 [[96  2]
 [ 9 90]]
Classification Report:
               precision    recall  f1-score   support

           0       0.91      0.98      0.95        98
           1       0.98      0.91      0.94        99

    accuracy                           0.94       197
   macro avg       0.95      0.94      0.94       197
weighted avg       0.95      0.94      0.94       197
