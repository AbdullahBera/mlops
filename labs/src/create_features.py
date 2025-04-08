import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import os
import yaml

params = yaml.safe_load(open("params.yaml"))["features"]
chi2percentile = params["chi2percentile"]

# File paths
train_path = "data/adult.data"
test_path = "data/adult.test"
output_dir = "data"
pipeline_path = os.path.join(output_dir, "pipeline.pkl")
train_output_path = os.path.join(output_dir, "processed_train_data.csv")
test_output_path = os.path.join(output_dir, "processed_test_data.csv")

# Column names (based on UCI Adult dataset)
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

# Load data
train = pd.read_csv(train_path, names=columns, na_values=" ?", skipinitialspace=True)
test = pd.read_csv(test_path, names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)

# Split features and target
X_train = train.drop("income", axis=1)
y_train = train["income"]
X_test = test.drop("income", axis=1)
y_test = test["income"]

# Preprocessing pipeline
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Fit and transform
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Save transformed data
pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed)\
    .to_csv(train_output_path, index=False)

pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed)\
    .to_csv(test_output_path, index=False)

# Save pipeline
joblib.dump(pipeline, pipeline_path)

print("Feature engineering complete!")
