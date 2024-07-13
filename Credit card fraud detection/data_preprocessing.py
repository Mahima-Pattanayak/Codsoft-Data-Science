import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data():
    data = pd.read_csv('creditcard.csv')
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    return preprocess_data(X, y)

def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    return X_train, X_test, y_train, y_test
