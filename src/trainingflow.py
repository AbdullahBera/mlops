from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_processing import preprocess_data

class TrainingFlow(FlowSpec):
    # Define parameters
    data_path = Parameter('data_path', default='cardio_train.csv', help='Path to training data')
    n_estimators = Parameter('n_estimators', default=100, help='Number of trees in RF')
    max_depth = Parameter('max_depth', default=10, help='Max depth of trees')
    test_size = Parameter('test_size', default=0.2, help='Test split size')
    random_state = Parameter('random_state', default=42, help='Random seed')

    @step
    def start(self):
        """Load and prepare the data"""
        # Load data
        heart_df = pd.read_csv(self.data_path, delimiter=";")
        
        # Preprocess data
        heart_df = preprocess_data(heart_df, is_training=True)
        
        # Split features and target
        X = heart_df.drop('cardio', axis=1)
        y = heart_df['cardio']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Save test set for later scoring demonstration
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        test_data.to_csv('test_data.csv', index=False)
        
        # Save column names for scoring
        self.feature_columns = list(X.columns)
        print("Data loaded and split successfully. Test data saved to test_data.csv")
        
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the model"""
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

    @step
    def register_model(self):
        """Register the model with MLflow"""
        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment('heart-disease-prediction')

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
                registered_model_name='heart-disease-model'
            )
        
        self.next(self.end)

    @step
    def end(self):
        """End the flow"""
        print(f"Training completed with accuracy: {self.accuracy}")
        print("Feature importances:")
        for feature, importance in zip(self.X_train.columns, self.model.feature_importances_):
            print(f"{feature}: {importance:.4f}")

if __name__ == '__main__':
    TrainingFlow()