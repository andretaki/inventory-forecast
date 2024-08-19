import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

class InventoryForecastModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = None

    def configure_model(self, training):
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,  # forecast horizon
            loss=pytorch_forecasting.metrics.MAE(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.model.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def create_model(training_data):
    model = InventoryForecastModel()
    model.configure_model(training_data)
    return model

def optimize_model(training_data):
    # Hyperparameter optimization
    study = optimize_hyperparameters(
        train_dataloaders=[training_data],
        model_path="tmp",
        n_trials=20,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )
    return study
