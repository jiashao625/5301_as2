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

# Remove rows with non-positive values in the 'Goal' column
df = df[df['Goal'] > 0].copy()  # Ensure no negative or zero goals
df['log_goal'] = np.log(df['Goal'])

# Remove any rows with NaN or infinite values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Define features
features = ['log_goal', 'duration_days', 'name_length', 'Category', 'Country']
X = df[features]
y = df['success']

# Streamlit app title
st.title('Kickstarter Campaign Success Prediction')

# Sidebar for feature selection
st.sidebar.header("Feature Selection")
selected_features = st.sidebar.multiselect("Select Features", features, default=features)

# Sidebar for number of cross-validation folds
cv_folds = st.sidebar.slider("Number of Cross-Validation Folds", min_value=2, max_value=10, value=5)

# Classifier Selection
st.header("Classifier Selection")
classifier_options = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(max_samples=0.1, random_state=42),  # 10% of the dataset
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

selected_classifier = st.selectbox('Select Classifier', list(classifier_options.keys()))

# Validate selected features
if not selected_features:
    st.error("Please select at least one feature.")
    st.stop()

# Dynamically determine numeric and categorical features based on selection
numeric_features = [f for f in selected_features if f in ['log_goal', 'duration_days', 'name_length']]
categorical_features = [f for f in selected_features if f in ['Category', 'Country']]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Combine preprocessing and classifier into a single pipeline
classifier = classifier_options[selected_classifier]
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', classifier)])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# Evaluate Model button
if st.button('Evaluate Model'):
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
