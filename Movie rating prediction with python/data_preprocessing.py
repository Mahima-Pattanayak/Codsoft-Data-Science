import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce

def load_data():
    # Load dataset
    data = pd.read_csv('movies.csv')
    X = data.drop(['rating'], axis=1)  # Adjust column name as needed
    y = data['rating']
    return preprocess_data(X, y)

def preprocess_data(X, y):
    # Encode categorical features
    encoder = ce.OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
