from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
from data_processing import preprocess_data

class ScoringFlow(FlowSpec):
    input_data = Parameter('input_data', help='Path to input data file')

    @step
    def start(self):
        """Load and preprocess the data to score"""
        # Load data
        self.raw_data = pd.read_csv(self.input_data)
        
        # Preprocess data
        self.data = preprocess_data(self.raw_data, is_training=False)
        print("Data loaded and preprocessed successfully")
        self.next(self.load_model)

    @step
    def load_model(self):
        """Load the registered model from MLflow"""
        mlflow.set_tracking_uri('http://localhost:5001')
        
        # Get the run that created the model to access parameters
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name('heart-disease-prediction')
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"]
            )
            if runs:
                self.feature_columns = eval(runs[0].data.params.get('feature_columns', '[]'))
        
        # Load the latest version of the model
        self.model = mlflow.sklearn.load_model(
            model_uri="models:/heart-disease-model/latest"
        )
        
        # Ensure data has all required columns in correct order
        self.data = self.data[self.feature_columns]
        self.next(self.make_predictions)

    @step
    def make_predictions(self):
        """Make predictions using the loaded model"""
        self.predictions = self.model.predict(self.data)
        print("Predictions generated successfully")
        self.next(self.end)

    @step
    def end(self):
        """Save predictions"""
        predictions_df = pd.DataFrame({
            'prediction': self.predictions
        })
        # Add original data to predictions
        result_df = pd.concat([self.raw_data, predictions_df], axis=1)
        result_df.to_csv('predictions.csv', index=False)
        print("Predictions saved to predictions.csv")

if __name__ == '__main__':
    ScoringFlow()