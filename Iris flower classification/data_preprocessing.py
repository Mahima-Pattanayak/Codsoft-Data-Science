import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    # Load dataset
    data = pd.read_csv('iris.csv')
    X = data.drop(['species'], axis=1)  # Adjust column name as needed
    y = data['species']
    return preprocess_data(X, y)

def preprocess_data(X, y):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
