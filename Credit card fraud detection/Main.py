# main.py

from data_preprocessing import load_data, clean_data, normalize_data
from model import train_model, evaluate_model
import utils

def main():
    # Load and preprocess data
    data = load_data('creditcard.csv')
    data = clean_data(data)
    data = normalize_data(data)
    
    # Split data, train model, and evaluate
    X_train, X_test, y_train, y_test = utils.train_test_split(data)
    model = train_model(X_train, y_train)
    performance = evaluate_model(model, X_test, y_test)
    
    # Display results or save model, etc.
    utils.plot_results(performance)

if __name__ == "__main__":
    main()
