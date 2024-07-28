import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import lightgbm as lgb

from config import Config
config = Config()

from data_handler import DataLoader, DataSplitter, DataPreprocessor
from model.lightgbm_model import LightGBMModelBuilder

def main():

    # Step 1: Load and preprocess the data
    data_loader = DataLoader(config.filename)
    df = data_loader.load_data()

    # Add indexes
    df["country_index"] = df["country"]
    df = df.set_index(["country_index", "year"])
    
    # Split data into train and test sets
    data_splitter = DataSplitter()
    train_df, valid_df, test_df = data_splitter.split_data(df, "year", "country_index")

    # Preprocess data
    data_preprocessor = DataPreprocessor()
    train_cat = train_df[['country']]
    valid_cat = valid_df[['country']]
    test_cat = test_df[['country']]
    train_num = train_df.drop(columns=['country'])
    valid_num = valid_df.drop(columns=['country'])
    test_num = test_df.drop(columns=['country'])

    # Preprocess categorical data using LabelEncoder
    train_cat_encoded, valid_cat_encoded, test_cat_encoded = data_preprocessor.preprocess_categorical_data(train_cat, valid_cat, test_cat)

    # Preprocess numerical data
    train_num_scaled, valid_num_scaled, test_num_scaled = data_preprocessor.preprocess_numerical_data(train_num, valid_num, test_num)

    # Concatenate categorical and numerical data
    train_combined, valid_combined, test_combined = data_preprocessor.concatenate_data(train_cat_encoded, valid_cat_encoded, test_cat_encoded, train_num_scaled, valid_num_scaled, test_num_scaled)
    X_train = train_combined.iloc[:, :-1]
    y_train = train_combined.iloc[:, -1]
    X_valid = valid_combined.iloc[:, :-1]
    y_valid = valid_combined.iloc[:, -1]
    X_test = test_combined.iloc[:, :-1]
    y_test = test_combined.iloc[:, -1]


    # Step 3: Build and train the LightGBM model
    lightgbm_builder = LightGBMModelBuilder(
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth
    )
    model = lightgbm_builder.build_model()
    early_stopping_callback = lgb.early_stopping(stopping_rounds=10)
    logging_callback = lgb.log_evaluation(period=1)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[early_stopping_callback, logging_callback])

    # Step 4: Make predictions
    valid_predictions = model.predict(X_valid)
    test_predictions = model.predict(X_test)

    # Step 5: Evaluate the model
    valid_mse = mean_squared_error(y_valid, valid_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    print(f'Validation MSE: {valid_mse}')
    print(f'Test MSE: {test_mse}')

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_train)), y_train, label='Training Data')
    plt.plot(range(len(y_train), len(y_train) + len(y_valid)), y_valid, label='Validation Data')
    plt.plot(range(len(y_train), len(y_train) + len(y_valid)), valid_predictions, label='Validation Predictions', linestyle='--')
    plt.plot(range(len(y_train) + len(y_valid), len(y_train) + len(y_valid) + len(y_test)), y_test, label='Test Data')
    plt.plot(range(len(y_train) + len(y_valid), len(y_train) + len(y_valid) + len(y_test)), test_predictions, label='Test Predictions', linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()