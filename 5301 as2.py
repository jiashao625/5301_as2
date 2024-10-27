import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('kickstarter_2016.csv')

# Data Preparation
df['success'] = df['State'].apply(lambda x: 1 if x.lower() == 'successful' else 0)
df['Launched'] = pd.to_datetime(df['Launched'])
df['Deadline'] = pd.to_datetime(df['Deadline'])
df['duration_days'] = (df['Deadline'] - df['Launched']).dt.days
df['name_length'] = df['Name'].apply(lambda x: len(x.split()))

# Check for non-positive values in the 'Goal' column
if (df['Goal'] <= 0).any():
    df = df[df['Goal'] > 0]  # Remove rows with non-positive values

# Create the log of the funding goal
df['log_goal'] = np.log(df['Goal'])

# Check for infinite values after log transformation
if np.isinf(df['log_goal']).any():
    st.error("The 'log_goal' contains infinite values after transformation. Please check the input data.")

# Remove any rows with NaN or infinite values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Select relevant features
features = ['log_goal', 'duration_days', 'name_length', 'Category', 'Country']
X = df[features]
y = df['success']

# Streamlit app title
st.title('Kickstarter Campaign Success Prediction')

# Sidebar for classifier selection
st.sidebar.header("Classifier Selection")
classifier_options = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(max_samples=0.1),  # 10% of the dataset
    'Gradient Boosting': GradientBoostingClassifier()
}
selected_classifier = st.sidebar.selectbox('Select Classifier', list(classifier_options.keys()))

# Sidebar for feature selection
st.sidebar.header("Feature Selection")
selected_features = st.sidebar.multiselect("Select Features", features, default=features)

# Sidebar for number of cross-validation folds
st.sidebar.header("Cross_Validation")
cv_folds = st.sidebar.slider("Number of Cross-Validation Folds", min_value=2, max_value=10, value=5)

# Preprocessing pipeline
numeric_features = ['log_goal', 'duration_days', 'name_length']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Category', 'Country']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding for categorical features
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Combine preprocessing and classifier into a single pipeline
classifier = classifier_options[selected_classifier]
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', classifier)])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# Cross-validation
if st.sidebar.button('Evaluate Model'):
    # Perform cross-validation
    accuracy_scores = cross_val_score(model_pipeline, X[selected_features], y, cv=cv_folds, scoring='accuracy')
    precision_scores = cross_val_score(model_pipeline, X[selected_features], y, cv=cv_folds, scoring='precision')
    recall_scores = cross_val_score(model_pipeline, X[selected_features], y, cv=cv_folds, scoring='recall')
    f1_scores = cross_val_score(model_pipeline, X[selected_features], y, cv=cv_folds, scoring='f1')

    # Display cross-validation results
    st.subheader("Cross-Validation Results")
    st.write(f"Accuracy: {accuracy_scores.mean():.2f}")
    st.write(f"Precision: {precision_scores.mean():.2f}")
    st.write(f"Recall: {recall_scores.mean():.2f}")
    st.write(f"F1 Score: {f1_scores.mean():.2f}")

    # Train the model on the entire training set for evaluation
    model_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = model_pipeline.predict(X_test)

    # Calculate evaluation metrics on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Display test set evaluation metrics
    st.subheader("Evaluation on Test Set")
    st.write(f"Accuracy: {test_accuracy:.2f}")
    st.write(f"Precision: {test_precision:.2f}")
    st.write(f"Recall: {test_recall:.2f}")
    st.write(f"F1 Score: {test_f1:.2f}")

