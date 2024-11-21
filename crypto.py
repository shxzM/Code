import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.impute import SimpleImputer
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense
import pandas as pd
import sys
import os

np.random.seed(7)

# Create necessary folders
for folder in ["models", "training_plot", "prediction_plot"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Enhanced preprocessing functions
def handle_missing_data(df):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Use linear interpolation instead of time-weighted
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = df[numerical_columns].interpolate(method='linear')
    
    # Use SimpleImputer for any remaining missing values
    imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    
    return df


def detect_and_remove_outliers(df, columns, threshold=3):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def create_technical_indicators(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    
    return df.dropna()

def graph(asset_name, pred, expected, plot_type='train'):
    plt.figure()
    plt.plot(expected, label='True Value')
    plt.plot(pred, label='Predicted Value')
    plt.title(f'Prediction by Model for {asset_name}')
    plt.xlabel('Time Scale')
    plt.ylabel('USD (Original Scale)')
    plt.legend()
    
    if plot_type == 'train':
        output_file = f"training_plot/{asset_name}_prediction.png"
    else:
        output_file = f"prediction_plot/{asset_name}_prediction.png"
    
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
    plt.close()

# Load and preprocess the dataset
df = pd.read_csv(
    'data/supplemental_train.csv',
    na_values=['null'],
    index_col='timestamp',
    parse_dates=True
)

# Load asset details
asset_details = pd.read_csv('data/asset_details.csv')
assets = asset_details[['Asset_ID', 'Asset_Name']].set_index('Asset_ID').to_dict()['Asset_Name']

# Define the features to be used
features = [
    'Count', 'Open', 'High', 'Low', 'Volume', 'VWAP', 
    'Target', 'MA7', 'MA30', 'RSI', 
    'Close_Lag1', 'Close_Lag2', 'Close_Lag3'
]

# Check for command-line arguments
if len(sys.argv) > 1:
    if sys.argv[1] == "train":
        # Enhanced preprocessing steps
        df = handle_missing_data(df)
        df = detect_and_remove_outliers(df, ['Volume', 'VWAP', 'Close'])
        df = create_technical_indicators(df)

        # Iterate through all assets
        for asset_id, asset_name in assets.items():
            asset_df = df[df['Asset_ID'] == asset_id]
            
            if asset_df.empty:
                print(f"No data for {asset_name}")
                continue

            print(f"\nTraining model for {asset_name} (Asset ID: {asset_id})")

            # Prepare features and target
            feature_scaler = MinMaxScaler()
            output_scaler = MinMaxScaler()

            # Additional check to ensure enough data
            if len(asset_df) < 100:
                print(f"Insufficient data for {asset_name}")
                continue

            # Prepare features and target
            feature_transform = feature_scaler.fit_transform(asset_df[features])
            feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=asset_df.index)

            output_var = asset_df['Close'].values.reshape(-1, 1)
            output_scaled = output_scaler.fit_transform(output_var)

            # Splitting the dataset using K-Fold
            kf = KFold(n_splits=5, shuffle=False)

            for fold, (train_index, test_index) in enumerate(kf.split(feature_transform)):
                print(f"Training fold {fold + 1}")
                
                # Train-test split
                X_train, X_test = feature_transform.iloc[train_index], feature_transform.iloc[test_index]
                y_train, y_test = output_scaled[train_index], output_scaled[test_index]
                
                # Reshape the data for LSTM input
                trainX = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
                testX = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
                
                # Build the LSTM model
                lstm = Sequential()
                lstm.add(LSTM(32, input_shape=(1, trainX.shape[2]), activation='relu', return_sequences=False))
                lstm.add(Dense(1))
                lstm.compile(loss='mean_squared_error', optimizer='adam')
                
                # Train the model
                lstm.fit(trainX, y_train, epochs=5, batch_size=8, verbose=1, shuffle=False)

                # Evaluate on test set
                pred = lstm.predict(testX)
                y_test_rescaled = output_scaler.inverse_transform(y_test)
                pred_rescaled = output_scaler.inverse_transform(pred)
                
                # Save predictions for the current fold
                graph(f"{asset_name}Fold{fold + 1}", pred_rescaled, y_test_rescaled)

            # Save the final model and weights for this asset
            model_json = lstm.to_json()
            with open(f"models/model_{asset_id}.json", "w") as json_file:
                json_file.write(model_json)
            lstm.save_weights(f"models/model_{asset_id}.weights.h5")
            print(f"Saved model for {asset_name} to disk")

        print("Training completed for all assets.")

    elif sys.argv[1] == "predict":
        # Prediction mode
        for asset_id, asset_name in assets.items():
            try:
                # Load the model for this asset
                json_file = open(f'models/model_{asset_id}.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                lstm = model_from_json(loaded_model_json)
                lstm.load_weights(f"models/model_{asset_id}.weights.h5")
                
                # Prepare data for prediction
                asset_df = df[df['Asset_ID'] == asset_id]
                
                if asset_df.empty:
                    print(f"No data for {asset_name}")
                    continue

                # Preprocessing
                asset_df = handle_missing_data(asset_df)
                asset_df = create_technical_indicators(asset_df)

                # Feature scaling
                feature_scaler = MinMaxScaler()
                output_scaler = MinMaxScaler()

                feature_transform = feature_scaler.fit_transform(asset_df[features])
                feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=asset_df.index)

                output_var = asset_df['Close'].values.reshape(-1, 1)
                output_scaled = output_scaler.fit_transform(output_var)

                # Time series split for prediction
                timesplit = TimeSeriesSplit(n_splits=10)
                for train_index, test_index in timesplit.split(feature_transform):
                    X_test = feature_transform.iloc[test_index]
                    y_test = output_scaled[test_index]
                
                testX = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
                pred = lstm.predict(testX)
                
                # Rescale predictions
                y_test_rescaled = output_scaler.inverse_transform(y_test)
                pred_rescaled = output_scaler.inverse_transform(pred)

                # Graph the predictions
                graph(asset_name, pred_rescaled, y_test_rescaled, plot_type='predict')

            except FileNotFoundError:
                print(f"No trained model found for {asset_name}")

    else:
        print("Invalid argument. Use 'train' or 'predict'.")
else:
    print("Please provide an argument: 'train' or 'predict'.")