from data_processing import load_data, preprocess_data
from model import train_model, evaluate_model
from evaluation import plot_results

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)

    # Plot results
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
