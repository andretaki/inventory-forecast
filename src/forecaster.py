import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE, MAPE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

class InventoryForecaster:
    def __init__(self, data_loader, forecast_horizon=30, context_length=60):
        self.data_loader = data_loader
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.model = None
        self.training = None
        self.validation = None

    def prepare_data(self):
        # Load and preprocess data
        orders = self.data_loader.load_orders()
        order_items = self.data_loader.load_order_items()
        data = self.data_loader.preprocess_data(orders, order_items)

        # Create features
        data['time_idx'] = (data['order_date'] - data['order_date'].min()).dt.days
        data['month'] = data['order_date'].dt.month
        data['year'] = data['order_date'].dt.year
        data['day_of_week'] = data['order_date'].dt.dayofweek

        # Split data into train and validation sets
        train_data = data[data['order_date'] < '2023-01-01']
        val_data = data[data['order_date'] >= '2023-01-01']

        # Create dataset
        self.training = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="quantity",
            group_ids=["product"],
            static_categoricals=["product"],
            time_varying_known_reals=["month", "year", "day_of_week"],
            time_varying_unknown_reals=["quantity"],
            max_encoder_length=self.context_length,
            max_prediction_length=self.forecast_horizon,
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, val_data, stop_randomization=True)

    def train_model(self, batch_size=64, max_epochs=100):
        # Create dataloaders
        train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size)
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=batch_size)

        # Configure the model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=1e-3,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=SMAPE(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )

        # Configure training
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,  # Set to number of GPUs if available
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_logger],
            logger=logger,
        )

        # Fit the model
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def make_predictions(self, start_date, end_date):
        # Load new data for prediction
        new_data = self.data_loader.load_orders(start_date, end_date)
        new_data = self.data_loader.preprocess_data(new_data, self.data_loader.load_order_items())

        # Prepare features
        new_data['time_idx'] = (new_data['order_date'] - new_data['order_date'].min()).dt.days
        new_data['month'] = new_data['order_date'].dt.month
        new_data['year'] = new_data['order_date'].dt.year
        new_data['day_of_week'] = new_data['order_date'].dt.dayofweek

        # Create a new dataset
        predict_dataset = TimeSeriesDataSet.from_dataset(self.training, new_data, stop_randomization=True)
        predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=128)

        # Make predictions
        predictions = self.model.predict(predict_dataloader)

        # Convert predictions to DataFrame
        pred_df = pd.DataFrame({
            'product': new_data['product'].unique(),
            'prediction': predictions.numpy().flatten()
        })

        return pred_df

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

