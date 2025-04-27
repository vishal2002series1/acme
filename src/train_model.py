import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Load the data
DATA_PATH = './data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_PATH = './models/churn_model.joblib'

def load_data(path):
    """Loads the data and preprocesses TotalCharges."""
    df = pd.read_csv(path)
    # Convert TotalCharges to numeric, handling errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)  # Drop rows with missing TotalCharges
    return df

def preprocess_data(df):
    """Preprocesses the data: encodes categoricals, scales numericals."""
    # Encode categorical features
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols.remove('customerID')  # Exclude customerID
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def train_model(df):
    """Trains a logistic regression model."""
    # Prepare data for modeling
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model

def save_model(model, path):
    """Saves the trained model to a file."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def main():
    """Main function to orchestrate data loading, preprocessing, training, and saving."""
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    model = train_model(df)
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()