from metaflow import FlowSpec, step, kubernetes, retry, timeout, catch, Parameter
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TrainingFlowGCP(FlowSpec):
    """
    A flow to train a model using GCP and Kubernetes
    """
    # Define parameters
    data_path = Parameter('data_path', default='cardio_train.csv', help='Path to training data')
    n_estimators = Parameter('n_estimators', default=100, help='Number of trees in RF')
    max_depth = Parameter('max_depth', default=10, help='Max depth of trees')
    test_size = Parameter('test_size', default=0.2, help='Test split size')
    random_state = Parameter('random_state', default=42, help='Random seed')
    
    @kubernetes
    @timeout(hours=2)
    @retry(times=2)
    @catch(var='error')
    @step
    def start(self):
        """
        Start the flow and load data
        """
        print("Starting the training flow in GCP...")
        
        # Load data
        heart_df = pd.read_csv(self.data_path, delimiter=";")
        
        # Split features and target
        X = heart_df.drop('cardio', axis=1)
        y = heart_df['cardio']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Save column names for scoring
        self.feature_columns = list(X.columns)
        print("Data loaded and split successfully")
        
        self.next(self.train_model)

    @kubernetes(cpu=2, memory=4000)  # Request more resources for training
    @timeout(hours=4)
    @retry(times=3)
    @step
    def train_model(self):
        """
        Train the model and log to MLFlow
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Get predictions and score
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Model trained with accuracy: {self.accuracy}")
        
        self.next(self.register_model)

    @kubernetes
    @timeout(hours=1)
    @retry(times=2)
    @step
    def register_model(self):
        """
        Register the model with MLflow
        """
        mlflow.set_tracking_uri('https://mlflow-server-148272679545.us-west2.run.app')
        mlflow.set_experiment('cardio_disease')

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'test_size': self.test_size,
                'feature_columns': self.feature_columns
            })
            
            # Log metrics
            mlflow.log_metric('accuracy', self.accuracy)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model, 
                'model',
                registered_model_name='heart-disease-model-gcp'
            )
        
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow
        """
        print(f"Training completed with accuracy: {self.accuracy}")
        print("Feature importances:")
        for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
            print(f"{feature}: {importance:.4f}")

if __name__ == '__main__':
    TrainingFlowGCP()