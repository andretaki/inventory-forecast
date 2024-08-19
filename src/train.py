import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE, MAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from data_loader import DataLoader
import torch

def train_model(train_dataloader, val_dataloader, max_epochs=100):
    # Create the TFT model
    tft = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,  # forecast horizon
        loss=SMAPE(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # Configure trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        limit_train_batches=50,
        enable_progress_bar=True
    )

    # Fit the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return tft, trainer

def evaluate_model(model, val_dataloader):
    predictions = model.predict(val_dataloader)
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    
    mae = MAE()(predictions, actuals)
    smape = SMAPE()(predictions, actuals)
    mape = MAPE()(predictions, actuals)

    print(f"MAE: {mae}")
    print(f"SMAPE: {smape}")
    print(f"MAPE: {mape}")

def main():
    # Initialize DataLoader
    data_loader = DataLoader()

    # Load and preprocess data
    orders = data_loader.load_orders('2022-01-01', '2024-12-31')
    order_items = data_loader.load_order_items()
    processed_data = data_loader.preprocess_data(orders, order_items)

    # Create datasets
    training, validation = data_loader.create_datasets(processed_data)

    # Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Train the model
    model, trainer = train_model(train_dataloader, val_dataloader)

    # Evaluate the model
    evaluate_model(model, val_dataloader)

    # Save the model
    trainer.save_checkpoint("tft_model.ckpt")

if __name__ == "__main__":
    main()
