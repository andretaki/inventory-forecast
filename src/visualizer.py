import matplotlib.pyplot as plt
from data_loader import DataLoader
from predict import load_model, make_predictions

def plot_predictions(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Values')
    plt.show()

def main():
    data_loader = DataLoader()
    model = load_model("tft_model.ckpt")
    
    # Load test data
    test_data = data_loader.load_test_data()  # You'll need to implement this method
    
    # Make predictions
    predictions = make_predictions(model, data_loader)
    
    # Plot results
    plot_predictions(test_data['target'], predictions)

if __name__ == "__main__":
    main()
