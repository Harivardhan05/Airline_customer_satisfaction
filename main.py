from src.preprocessing import preprocess_data, split_data
from src.model import train_model
from src.utils import evaluate_model, save_model

if __name__ == "__main__":
    # Preprocess the data
    data = preprocess_data('path/to/your/dataset.csv')

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, target_column='satisfaction')

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the trained model
    save_model(model, 'customer_satisfaction_model.pkl')
