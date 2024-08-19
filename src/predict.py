from pytorch_forecasting import TemporalFusionTransformer
from data_loader import DataLoader

def load_model(model_path):
    return TemporalFusionTransformer.load_from_checkpoint(model_path)

def make_predictions(model, data_loader):
    # Load your test data
    test_data = data_loader.load_test_data()  # You'll need to implement this method
    
    # Make predictions
    predictions = model.predict(test_data)
    
    return predictions

def main():
    model = load_model("tft_model.ckpt")
    data_loader = DataLoader()
    predictions = make_predictions(model, data_loader)
    print(predictions)

if __name__ == "__main__":
    main()
