import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
import pandas as pd
from urllib.parse import quote_plus
import logging
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE, MAPE
import pytorch_lightning as pl

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class DataLoader:
    def __init__(self):
        # Load database credentials from environment variables
        self.host = os.getenv('POSTGRES_HOST')
        self.database = os.getenv('POSTGRES_DATABASE')
        self.user = os.getenv('POSTGRES_USER')
        self.password = os.getenv('POSTGRES_PASSWORD')
        self.port = os.getenv('POSTGRES_PORT', '5432')

        # Safely encode the password
        encoded_password = quote_plus(self.password)
        self.connection_string = f"postgresql://{self.user}:{encoded_password}@{self.host}:{self.port}/{self.database}"
        self.engine = create_engine(self.connection_string)

    def test_connection(self):
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                logging.info("Successfully connected to the database.")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to the database: {str(e)}")
            return False

    def load_orders(self, start_date, end_date):
        query = """
        SELECT * FROM orders
        WHERE order_date BETWEEN :start_date AND :end_date
        """
        df = pd.read_sql_query(text(query), self.engine, params={'start_date': start_date, 'end_date': end_date})
        logging.info(f"Loaded orders. Shape: {df.shape}")
        if not df.empty:
            logging.info(f"Order date range: from {df['order_date'].min()} to {df['order_date'].max()}")
            logging.info(f"Number of unique order dates: {df['order_date'].nunique()}")
        else:
            logging.warning("No orders found in the specified date range.")
        return df

    def load_order_items(self):
        query = "SELECT * FROM order_items"
        df = pd.read_sql_query(text(query), self.engine)
        logging.info(f"Loaded order items. Shape: {df.shape}")
        return df

    def load_addresses(self):
        query = "SELECT * FROM addresses"
        df = pd.read_sql_query(text(query), self.engine)
        logging.info(f"Loaded addresses. Shape: {df.shape}")
        return df

    def load_failed_orders(self):
        query = "SELECT * FROM failed_orders"
        df = pd.read_sql_query(text(query), self.engine)
        logging.info(f"Loaded failed orders. Shape: {df.shape}")
        return df

    def preprocess_data(self, orders_df, order_items_df):
        if orders_df.empty or order_items_df.empty:
            logging.warning("Orders or order items DataFrame is empty. Cannot process data.")
            return pd.DataFrame()

        logging.info(f"Preprocessing data. Orders shape: {orders_df.shape}, Order items shape: {order_items_df.shape}")
        
        # Merge orders with order_items
        df = pd.merge(orders_df, order_items_df, left_on='id', right_on='order_id', how='inner')
        logging.info(f"Merged data shape: {df.shape}")
        
        if df.empty:
            logging.warning("No matching orders and order items. Cannot process further.")
            return pd.DataFrame()

        # Convert order_date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Sort by date
        df = df.sort_values('order_date')
        
        # Group by date and product, sum the quantities
        df = df.groupby(['order_date', 'sku'])['quantity'].sum().reset_index()
        logging.info(f"Grouped data shape: {df.shape}")
        
        # Pivot the table to have products as columns
        df_pivoted = df.pivot(index='order_date', columns='sku', values='quantity').fillna(0)
        logging.info(f"Pivoted data shape: {df_pivoted.shape}")
        
        # Resample to daily frequency and forward fill missing values
        df_daily = df_pivoted.resample('D').ffill()
        logging.info(f"Final daily data shape: {df_daily.shape}")
        
        # Additional preprocessing for Temporal Fusion Transformer
        df_daily = df_daily.reset_index()
        df_daily['time_idx'] = (df_daily['order_date'] - df_daily['order_date'].min()).dt.days
        
        # Melt the dataframe to long format
        df_long = df_daily.melt(id_vars=['order_date', 'time_idx'], 
                                var_name='product', 
                                value_name='quantity')
        
        # Add any static covariates here if available
        # df_long['static_covariate'] = ...
        
        # Add any known future information here if available
        # df_long['future_info'] = ...
        
        return df_long

    def create_datasets(self, df, product_col='product', target_col='quantity', 
                        time_idx_col='time_idx', static_categoricals=None, 
                        static_reals=None, time_varying_known_reals=None, 
                        time_varying_unknown_reals=None, max_encoder_length=30, 
                        max_prediction_length=7):
        
        if static_categoricals is None:
            static_categoricals = [product_col]
        if static_reals is None:
            static_reals = []
        if time_varying_known_reals is None:
            time_varying_known_reals = []
        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = [target_col]

        training_cutoff = df[time_idx_col].max() - max_prediction_length

        training = TimeSeriesDataSet(
            df[lambda x: x[time_idx_col] <= training_cutoff],
            time_idx=time_idx_col,
            target=target_col,
            group_ids=[product_col],
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
        )

        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

        return training, validation

    def get_table_names(self):
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_product_list(self):
        query = "SELECT DISTINCT sku FROM order_items ORDER BY sku"
        products = pd.read_sql_query(text(query), self.engine)['sku'].tolist()
        logging.info(f"Number of unique products: {len(products)}")
        return products

    def print_table_structure(self, table_name):
        query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        """
        structure = pd.read_sql_query(text(query), self.engine)
        logging.info(f"\n{table_name} table structure:\n{structure}")

if __name__ == "__main__":
    loader = DataLoader()
    
    if loader.test_connection():
        relevant_tables = ['addresses', 'failed_orders', 'orders', 'order_items']
        logging.info(f"Relevant tables: {[table for table in loader.get_table_names() if table in relevant_tables]}")
        
        # Print table structures
        for table in relevant_tables:
            loader.print_table_structure(table)
        
        # Load and preprocess data
        start_date = '2022-01-01'
        end_date = '2024-12-31'
        logging.info(f"Loading orders from {start_date} to {end_date}")
        orders = loader.load_orders(start_date, end_date)
        order_items = loader.load_order_items()
        
        if not orders.empty and not order_items.empty:
            processed_data = loader.preprocess_data(orders, order_items)
            
            if not processed_data.empty:
                logging.info(f"Processed data shape: {processed_data.shape}")
                logging.info(f"Processed data columns: {processed_data.columns}")
                
                # Create training and validation datasets
                training, validation = loader.create_datasets(processed_data)
                logging.info(f"Training dataset: {training}")
                logging.info(f"Validation dataset: {validation}")
            else:
                logging.warning("Processed data is empty.")
        else:
            logging.warning("Orders or order items data is empty. Cannot process further.")
        
        products = loader.get_product_list()
    else:
        logging.error("Failed to connect to the database. Please check your credentials and try again.")
