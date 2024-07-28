import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from data_handler import DataLoader, DataPreprocessor, DataSplitter, DataReshaperLSTM
from model.lstm_model import LSTMModelBuilder

def main():
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the LSTM model training process.")

    # Load config
    config = Config()
    logging.info("Configuration loaded.")

    # Load data
    logging.info("Loading data...")
    data_loader = DataLoader(config.filename)
    df = data_loader.load_data()
    logging.info("Data loaded successfully.")

    # Add indexes
    df["country_index"] = df["country"]
    df = df.set_index(["country_index", "year"])
    logging.info("Index set to 'country' and 'year'.")

    #Iran data to check on the graph
    iran_co2 = df[df['country'] == 'Iran']['co2_including_luc']
    iran_co2 = iran_co2.reset_index(level='country_index', drop=True)
    iran_co2.plot(title='CO2 Including LUC for Iran', xlabel='Index', ylabel='CO2 Including LUC')
    plt.grid(True)
    plt.xticks(ticks=iran_co2.index.values, labels=iran_co2.index.values.astype(int), rotation=45)
    plt.savefig('output/iran_co2.png')

    # Split data into train and test sets
    data_splitter = DataSplitter()
    train_df, valid_df, test_df = data_splitter.split_data(df, "year", "country_index")
    train_df.to_csv('output/1_split_data_train_df.csv')
    valid_df.to_csv('output/1_split_data_valid_df.csv')
    test_df.to_csv('output/1_split_data_test_df.csv')
    logging.info("Data split into training, validation and testing sets.")

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
    logging.info("Categorical data preprocessed.")

    # Preprocess numerical data
    train_num_scaled, valid_num_scaled, test_num_scaled = data_preprocessor.preprocess_numerical_data(train_num, valid_num, test_num)
    logging.info("Numerical data preprocessed.")

    # Concatenate categorical and numerical data
    train_combined, valid_combined, test_combined = data_preprocessor.concatenate_data(train_cat_encoded, valid_cat_encoded, test_cat_encoded, train_num_scaled, valid_num_scaled, test_num_scaled)
    train_combined.to_csv('output/2_preprocess_train_combined.csv')
    valid_combined.to_csv('output/2_preprocess_valid_combined.csv')
    test_combined.to_csv('output/2_preprocess_test_combined.csv')
    logging.info("Categorical and numerical data concatenated.")

    # Reshape data for LSTM
    data_resherper = DataReshaperLSTM()
    x_train, x_val, x_test, y_train, y_val, y_test, train_idx, valid_idx, test_idx = data_resherper.reshape_data(train_combined, valid_combined, test_combined)

    x_train_str = np.array2string(x_train, separator=', ')
    with open('output/4_reshape_x_train.csv', 'w') as f:
        f.write(x_train_str)
    
    x_test_str = np.array2string(x_test, separator=', ')
    with open('output/4_reshape_x_test.csv', 'w') as f:
        f.write(x_test_str)

    x_val_str = np.array2string(x_val, separator=', ')
    with open('output/4_reshape_x_val.csv', 'w') as f:
        f.write(x_val_str)
    
    y_val_str = np.array2string(y_val, separator=', ')
    with open('output/4_reshape_y_val.csv', 'w') as f:
        f.write(y_val_str)

    y_train_str = np.array2string(y_train, separator=', ')
    with open('output/4_reshape_y_train.csv', 'w') as f:
        f.write(y_train_str)

    y_test_str = np.array2string(y_test, separator=', ')
    with open('output/4_reshape_y_test.csv', 'w') as f:
        f.write(y_test_str)    

    logging.info("Data reshaped for LSTM model.")

    # Build LSTM model
    logging.info("Building LSTM model.")
    lstm_model_builder = LSTMModelBuilder(
        input_shape=(x_train.shape[1], x_train.shape[2])
    )
    model = lstm_model_builder.build_model()
    model.summary()
    logging.info("LSTM model built successfully.")

    # Train the model
    logging.info("Starting model training.")
    history = model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size, validation_data = (x_val, y_val))
    logging.info("Model training completed.")

    # Save the model
    model.save('output/lstm_model.h5')
    logging.info("Model saved to 'output/lstm_model.h5'.")

    # Make predictions
    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)
    logging.info("Predictions made.")

    inverted_data_predicted_train_y = data_preprocessor.inverse_transform_data(train_predictions, train_predictions.shape[0], train_num_scaled.shape[1])
    inverted_data_train_y = data_preprocessor.inverse_transform_data(y_train, train_predictions.shape[0], train_num_scaled.shape[1])

    inverted_data_predicted_val_y = data_preprocessor.inverse_transform_data(val_predictions, val_predictions.shape[0], valid_num_scaled.shape[1])
    inverted_data_val_y = data_preprocessor.inverse_transform_data(y_val, val_predictions.shape[0], valid_num_scaled.shape[1])

    inverted_data_predicted_test_y = data_preprocessor.inverse_transform_data(test_predictions, test_predictions.shape[0], test_num_scaled.shape[1])
    inverted_data_test_y = data_preprocessor.inverse_transform_data(y_test, test_predictions.shape[0], test_num_scaled.shape[1])

    train_predictions = pd.DataFrame(data = {'actual': inverted_data_train_y[:, -1].reshape(-1), 'predicted': inverted_data_predicted_train_y[:, -1].reshape(-1)}, index = train_idx)
    valid_predictions = pd.DataFrame(data = {'actual': inverted_data_val_y[:, -1].reshape(-1), 'predicted': inverted_data_predicted_val_y[:, -1].reshape(-1)}, index = valid_idx)
    test_predictions = pd.DataFrame(data = {'actual': inverted_data_test_y[:, -1].reshape(-1), 'predicted': inverted_data_predicted_test_y[:, -1].reshape(-1)}, index = test_idx)

    # Plot actual vs predictions for each country in both training and test data
    logging.info("Creating plots.")
    countries = df.index.get_level_values('country_index').unique()
    num_countries = len(countries)
    fig, axes = plt.subplots(nrows=num_countries, ncols=3, figsize=(15, num_countries * 5))

    for i, country in enumerate(countries):
        print(f"Processing country: {country}, subplot index: {i}")
        
        # Plot training data
        ax = axes[i][0]
        if country in train_predictions.index.get_level_values('country_index'):
            train_data = train_predictions.loc[country]
            if isinstance(train_data, pd.DataFrame) and not train_data.empty:
                ax.plot(train_data.index.get_level_values('year'), train_data['actual'], label='Train Actual', color='blue', marker='o')
                ax.plot(train_data.index.get_level_values('year'), train_data['predicted'], label='Train Predicted', linestyle='--', color='blue', marker='o')
                ax.set_title(f'{country} - Train')
                ax.legend()
                ax.grid(True)
                ax.set_xticks(train_data.index.get_level_values('year'))  # Set x-ticks to years
                ax.set_xticklabels(train_data.index.get_level_values('year').astype(int), rotation=45)  # Format x-ticks as integers


        # Plot validation data
        ax = axes[i][1]
        if country in valid_predictions.index.get_level_values('country_index'):
            valid_data = valid_predictions.loc[country]
            if isinstance(valid_data, pd.DataFrame) and not valid_data.empty:
                ax.plot(valid_data.index.get_level_values('year'), valid_data['actual'], label='Validation Actual', color='orange', marker='o')
                ax.plot(valid_data.index.get_level_values('year'), valid_data['predicted'], label='Validation Predicted', linestyle='--', color='orange', marker='o')
                ax.set_title(f'{country} - Validation')
                ax.legend()
                ax.grid(True)
                ax.set_xticks(valid_data.index.get_level_values('year'))  # Set x-ticks to years
                ax.set_xticklabels(valid_data.index.get_level_values('year').astype(int), rotation=45)  # Format x-ticks as integers


        # Plot test data
        ax = axes[i][2]
        if country in test_predictions.index.get_level_values('country_index'):
            test_data = test_predictions.loc[country]
            if isinstance(test_data, pd.DataFrame) and not test_data.empty:
                ax.plot(test_data.index.get_level_values('year'), test_data['actual'], label='Test Actual', color='green', marker='o')
                ax.plot(test_data.index.get_level_values('year'), test_data['predicted'], label='Test Predicted', linestyle='--', color='green', marker='o')
                ax.set_title(f'{country} - Test')
                ax.legend()
                ax.grid(True)
                ax.set_xticks(test_data.index.get_level_values('year'))  # Set x-ticks to years
                ax.set_xticklabels(test_data.index.get_level_values('year').astype(int), rotation=45)  # Format x-ticks as integers


    plt.tight_layout()
    plt.savefig('output/combined_actual_vs_predicted.png')
    #plt.show()
    logging.info("Combined training and test data plots created and saved to 'output/combined_actual_vs_predicted.png'.")

    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_test_pred)
    # rmse = sqrt(mse)
    # log_metric("rmse", rmse)

    # # Save and log the model
    # model.save("model.h5")
    # run["model"].upload("model.h5")

    # # Stop the Neptune run
    # run.stop()


if __name__ == "__main__":
    main()
