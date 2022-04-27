import numpy as np
import pandas as pd
from pickle import dump

from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

from sklearn.preprocessing import MinMaxScaler

from dagster import op, Out

@op(out={"dfe": Out(), "dfuv": Out(), "dftemp": Out()})
def create_dataframe(context, dfe_path:str, dfuv_path:str, dftemp_path:str):

    dfe = pd.read_csv(dfe_path, parse_dates=['timestamp'],
                      index_col='timestamp')
    dfuv = pd.read_csv(dfuv_path, parse_dates=['timestamp'],
                       index_col='timestamp')
    dftemp = pd.read_csv(dftemp_path, parse_dates=['timestamp'],
                         index_col='timestamp')

    # logger = get_dagster_logger()
    # logger.info(f"raining set score: {train_score}")
    # logger.info(f"Test set score: {test_score}")

    return dfe, dfuv, dftemp

@op
def merge_all_data(context, dfe: pd.DataFrame, dfuv: pd.DataFrame, dftemp: pd.DataFrame):
    dfe = dfe.resample('5MIN').mean()
    dfuv = dfuv['uv_index'].resample('5min').ffill()

    df = pd.merge_asof(dfe, dftemp, on='timestamp')
    df = pd.merge_asof(df, dfuv, on='timestamp')

    return df

@op(out={"features_df": Out(), "day_df": Out()})
def build_feature_frame(context, dfe):
    """Build Feature DataFrame for forecasting"""

    # Step 1 feature engineering
    dfe['prev_power_solar'] = dfe.shift(1)['power_solar']
    dfe['irradiance_1daybefore'] = dfe['total_irradiance'].shift(periods=+288)
    dfe["power_change"] = dfe.apply(
        lambda row: 0 if np.isnan(row['prev_power_solar']) else row['power_solar'] - row['prev_power_solar'], axis=1)
    dfe['is_weekday'] = dfe.apply(
        lambda row: 1 if row['timestamp'].dayofweek <= 4 else 0, axis=1)
    dfe['is_daytime'] = dfe.apply(
        lambda row: 1 if (row['timestamp'].hour >= 5) & (row['timestamp'].hour <= 19) else 0, axis=1)

    # Step 2 iterate each row to built feature df
    rows = []
    for _, row in dfe.iterrows():  # iterate each row  with progress checking
        row_data = dict(
            power_solar=row['power_solar'],
            hour_of_day=row['timestamp'].hour,
            day_of_week=row['timestamp'].dayofweek,
            is_week_day=row['is_weekday'],
            is_daytime=row['is_daytime'],
            week_of_year=row['timestamp'].week,
            month=row['timestamp'].month,
            uv_index=row['uv_index'],
            wind_speed=row['wind_speed'],
            irradiance=row['total_irradiance'],
            ambient_temp=row['ambient_temperature'],
            module_temp=row['pv_module_temperature'],
            power_change=row['power_change'],
            irradiance_1daybefore=row['irradiance_1daybefore'],
        )
        rows.append(row_data)
    features_df = pd.DataFrame(rows)

    day_df = pd.DataFrame()
    day_df['day_of_week_sin'] = features_df['day_of_week'].mul(2 * np.pi / 7).apply(np.sin)
    day_df['day_of_week_cos'] = features_df['day_of_week'].mul(2 * np.pi / 7).apply(np.cos)
    features_df.drop(columns=['day_of_week'], inplace=True)

    return features_df, day_df

@op(out={"train_df": Out(), "test_df": Out(), "day_train": Out(), "day_test": Out()})
def split_data(context, features_df: pd.DataFrame, day_df: pd.DataFrame, fraction: float):
    train_size = int(len(features_df) * fraction)

    train_df, test_df = features_df[:train_size], features_df[train_size:]
    day_train, day_test = day_df[:train_size], day_df[train_size:]
    return train_df, test_df, day_train, day_test

@op
def train_scaler(context, train_df, scaler_name):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df)
    dump(scaler, open(f'{scaler_name}.pkl', 'wb'))

    return scaler

def Scale(df: pd.DataFrame, scaler: MinMaxScaler) :
    return pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

@op(out={"train_df": Out(), "test_df": Out()})
def preprocess_scaled(context, train_df, test_df, day_train, day_test, scaler):
    train_df = Scale(train_df, scaler)
    test_df = Scale(test_df, scaler)

    train_df = pd.concat([train_df, day_train], axis=1)
    test_df = pd.concat([test_df, day_test], axis=1)

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    return train_df, test_df

def split_feature(df: pd.DataFrame, pred_next: int = 12) -> tuple:
    X = df.iloc[:-pred_next].drop(columns=['power_solar'])
    y = df.iloc[pred_next:]['power_solar']
    return np.asarray(X), np.asarray(y)

@op(out={"X_train": Out(), "X_test": Out(), "y_train": Out(), "y_test": Out()})
def split_X_y(context, train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train, y_train = split_feature(train_df)
    X_test, y_test = split_feature(test_df)

    return X_train, X_test, y_train, y_test

def descale(descaler: MinMaxScaler, values):
    values_2D = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2D).flatten()

@op
def create_mlp_model(context, X_train, optimizer):
    K.clear_session()
    model = Sequential()
    model.add(Dropout(0.3, input_shape = (X_train.shape[1],)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=[RootMeanSquaredError()])
    return model

@op
def train_model(context, X_train, y_train, X_test, y_test, model, num_epochs, num_batch_size, model_name):
    history = model.fit(X_train, y_train,
                        epochs=num_epochs,
                        batch_size=num_batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
                        )

    model.save(f'{model_name}.h5')

    return model

@op
def prediction_unseen(context, model, scaler, newdata):
    y_pred = model.predict(newdata)
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[0]
    y_pred = y_pred.ravel()
    y_pred_descaled = descale(descaler, y_pred)
    return y_pred_descaled






