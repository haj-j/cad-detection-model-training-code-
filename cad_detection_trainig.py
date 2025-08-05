import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Configure display settings
pd.set_option('display.max_columns', 50)
plt.style.use('ggplot')

def load_and_validate_data(file_path):
    """Load and validate the dataset"""
    print(f"\n{'='*50}\nLoading data from: {file_path}\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully")
        print(f"Dataset shape: {df.shape}")
        
        # Validate required columns
        required_columns = {'Status', 'Sex'}  
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Handle data preprocessing"""
    print("\n" + "="*50)
    print("Preprocessing Data")
    print("="*50)
    
    # Separate features and target
    X = df.drop(columns='Status')
    Y = df['Status']
    
    # Encode target
    Y = Y.map({0: 'not sick', 1: 'sick'})
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    print(f"Target classes: {list(le.classes_)}")
    print(f"Class distribution:\n{pd.Series(Y).value_counts()}")
    
    # Encode categorical features
    if 'Sex' in X.columns:
        X['Sex'] = X['Sex'].map({'Male': 1, 'Female': 0})
        print("\nSex column encoded: Male=1, Female=0")
    
    return X, Y, le

def train_and_evaluate(X, Y, le):
    """Train model and evaluate performance"""
    print("\n" + "="*50)
    print("Training Model")
    print("="*50)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=Y
    )
    print(f"Train/test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Features scaled using StandardScaler")
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  
    )
    
    print("\nModel Parameters:")
    print(model.get_params())
    
    print("\nTraining model...")
    model.fit(X_train, Y_train)
    print("Model trained successfully")
    
    # Evaluate
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    for name, data, actual in [('Train', X_train, Y_train), ('Test', X_test, Y_test)]:
        pred = model.predict(data)
        proba = model.predict_proba(data)[:, 1]
        
        print(f"\n{name} Set Performance:")
        print(f"Accuracy: {accuracy_score(actual, pred):.4f}")
        print(f"AUC-ROC: {roc_auc_score(actual, proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(actual, pred, target_names=le.classes_))
    
    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(Y_test, model.predict(X_test))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return model, scaler

def save_artifacts(model, le, scaler):
    """Save model and preprocessing artifacts"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('model_artifacts', exist_ok=True)
    
    artifacts = {
        'model': f'model_artifacts/cad_model_{timestamp}.pkl',
        'encoder': f'model_artifacts/label_encoder_{timestamp}.pkl',
        'scaler': f'model_artifacts/scaler_{timestamp}.pkl'
    }
    
    joblib.dump(model, artifacts['model'])
    joblib.dump(le, artifacts['encoder'])
    joblib.dump(scaler, artifacts['scaler'])
    
    print("\nArtifacts saved:")
    for name, path in artifacts.items():
        print(f"{name:>8}: {path}")
    
    return artifacts

def main():
    try:
        # Configuration
        DATA_PATH = r"C:\Users\HP\Desktop\CaD File dataset 1\dataset.csv"
        
        # Pipeline
        df = load_and_validate_data(DATA_PATH)
        X, Y, le = preprocess_data(df)
        model, scaler = train_and_evaluate(X, Y, le)
        artifacts = save_artifacts(model, le, scaler)
        
        print("\nTraining completed successfully!")
        print("\nArtifact paths for prediction script:")
        print(f"ARTIFACT_PATHS = {artifacts}")
        
    except Exception as e:
        print(f"\nError in training pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting CAD Detection Model Training")
    main()
    
