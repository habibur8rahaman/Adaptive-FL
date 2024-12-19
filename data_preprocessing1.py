import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import pickle
import json


def preprocess_data(folder_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    global_features = set()

    global_scaler = None
    label_encoders = {}


    for file in os.listdir(folder_path):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, dtype=str)
            global_features.update(df.columns)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    mandatory_columns = {'date', 'time', 'label', 'type'}
    global_features = global_features - mandatory_columns
    global_features = sorted(global_features)


    feature_file = os.path.join(output_folder, "global_features.json")
    with open(feature_file, 'w') as f:
        json.dump(global_features, f)

    for file in os.listdir(folder_path):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, file)
        print(f"Processing file: {file}")

        try:

            df = pd.read_csv(file_path, dtype=str)

            if not mandatory_columns.issubset(df.columns):
                raise ValueError(f"Missing mandatory columns in {file}: {mandatory_columns - set(df.columns)}")

            df['timestamp'] = pd.to_datetime(
                df['date'].str.strip() + ' ' + df['time'].str.strip(),
                format='%d-%b-%y %H:%M:%S',
                errors='coerce'
            )

            df.dropna(subset=['timestamp'], inplace=True)

            df.drop(columns=['date', 'time'], inplace=True)

            for col in global_features:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[global_features + ['label', 'type', 'timestamp']]

            for col in global_features:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            if global_scaler is None:
                global_scaler = MinMaxScaler()
                df[global_features] = global_scaler.fit_transform(df[global_features])
            else:
                df[global_features] = global_scaler.transform(df[global_features])

            categorical_cols = ['label', 'type']
            for col in categorical_cols:
                if col not in label_encoders:
                    label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))

            oversampler = RandomOverSampler(random_state=42)
            X = df.drop(columns=['label', 'timestamp', 'type'])
            y = df['label']
            X_resampled, y_resampled = oversampler.fit_resample(X, y)

            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled['label'] = y_resampled.reset_index(drop=True)
            df_resampled['type'] = df['type'].iloc[:len(df_resampled)].reset_index(drop=True)
            df_resampled['timestamp'] = pd.to_datetime(df['timestamp']).iloc[:len(df_resampled)].reset_index(drop=True)

            output_file = os.path.join(output_folder, f"preprocessed_{file}")
            df_resampled.to_csv(output_file, index=False)
            print(f"File saved: {output_file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    with open(os.path.join(output_folder, 'scalers_and_encoders.pkl'), 'wb') as f:
        pickle.dump({'scaler': global_scaler, 'encoders': label_encoders}, f)

    print("Preprocessing complete. All files processed and saved.")


input_folder = "D:/research/FL/Adaptive FL/raw_data"
output_folder = "D:/research/FL/Adaptive FL/preprocessed_data"

preprocess_data(input_folder, output_folder)
