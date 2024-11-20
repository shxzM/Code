import pandas as pd
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

seed0 = 8008135
params = {
    # 'early_stopping_rounds': 50,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': 3,  # changed from 5
    'verbose': -1,
    'max_bin': 600,
    'min_data_in_leaf': 50,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'subsample_freq': 1,
    'feature_fraction': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 2,
    'seed': seed0,
    'feature_fraction_seed': seed0,
    'bagging_fraction_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'extra_trees': True,
    'extra_seed': seed0,
    'zero_as_missing': True,
    "first_metric_only": True
}

print("Loading the dataset...")
data = pd.read_csv('train.csv', dtype='float64')

len_list = []
rmse_list = []
test_list = []
predictions_list = []
assets = ['Binance Coin', 'Bitcoin', 'Bitcoin Cash', 'Cardano', 'Dogecoin', 'EOS.IO', 'Ethereum',
          'Ethereum Classic', 'IOTA', 'Litecoin', 'Maker', 'Monero', 'Stellar', 'TRON']

for asset_id in range(14):
    X = data.loc[data['Asset_ID'] == asset_id, 'Close'].values
    # X = 100000*X/(max(X) - min(X))
    X = X/max(X)

    # Initialize the model
    print(f'Initializing the model for {assets[asset_id]}...')
    model = None
    model_fit = None
    length = len(X)

    # Define the size of chunks
    chunk_size = int(length * 0.5)
    test_size = int(length * 0.25)
    num_chunks = length // (chunk_size + test_size)

    for i in range(num_chunks):
        # Divide the data into chunks
        train = X[i * (chunk_size + test_size):i * (chunk_size + test_size) + chunk_size]
        test = X[i * (chunk_size + test_size) + chunk_size:(i + 1) * (chunk_size + test_size)]

        print(f'Using chunk {i + 1}/{num_chunks + 1} for training {assets[asset_id]}...')
        if model_fit is not None:
            train = train.reshape(-1, 1)
            model_fit = model_fit.fit(train, train, init_score=model_fit.predict(train))
        else:
            model = LGBMRegressor(**params)
            train = train.reshape(-1, 1)
            model_fit = model.fit(train, train)

        print(f'Testing the {assets[asset_id]} model {i + 1}/{num_chunks + 1} on chunk {i + 2}/{num_chunks + 1}...')
        test = test.reshape(-1, 1)
        predictions = model_fit.predict(test)
        rmse = sqrt(mean_squared_error(test, predictions))/(max(test) - min(test))
        print(f'Test RMSE for {assets[asset_id]}: %.3f' % rmse)
        test_list += [test]
        predictions_list += [predictions]
        rmse_list += [rmse]
        len_list += [length]

# Plot actual vs predicted values
print("Plotting actual vs predicted values...")
for test, prediction, asset, rmse, length in zip(test_list, predictions_list, assets, rmse_list, len_list):
    print(f'Plotting {asset}...')
    plt.figure(figsize=(10, 8))
    plt.plot(test, label=f'{asset} Actual')
    plt.plot(prediction, label=f'{asset} Predicted')
    plt.xlabel('Time steps')
    plt.ylabel('Close Price')
    plt.title(f'Actual vs Predicted Close Prices for {asset} ({length} data points) RMSE = %.3f' % rmse)
    plt.legend()
    # plt.show()
    plt.savefig(f'lightGBM-chunked-%.3f-{asset}.png' % rmse)