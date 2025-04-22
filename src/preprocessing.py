import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import os

# Load data
train = pd.read_csv("data/adult.data", header=None, na_values=" ?", skipinitialspace=True)
test = pd.read_csv("data/adult.test", header=None, skiprows=1, na_values=" ?", skipinitialspace=True)

# Add column names (UCI Adult dataset)
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
train.columns = columns
test.columns = columns

X_train = train.drop("income", axis=1)
y_train = train["income"]
X_test = test.drop("income", axis=1)
y_test = test["income"]

# Build pipeline
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor)
])

# Transform data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Save results
os.makedirs("data", exist_ok=True)
pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed)\
    .to_csv("data/processed_train_data.csv", index=False)

pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed)\
    .to_csv("data/processed_test_data.csv", index=False)

joblib.dump(pipeline, "data/pipeline.pkl")
