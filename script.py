from modules import functions as fn
from dagster import execute_pipeline, pipeline

@pipeline
def data_pipeline():
    dfe, dfuv, dftemp = fn.create_dataframe()
    df = fn.merge_all_data(dfe, dfuv, dftemp)
    features_df, day_df = fn.build_feature_frame(df)

    train_df, test_df, day_train, day_test = fn.split_data(features_df = features_df, day_df = day_df)
    scaler = fn.train_scaler(train_df = train_df)
    train_df, test_df = fn.preprocess_scaled(train_df, test_df, day_train, day_test, scaler)
    X_train, X_test, y_train, y_test = fn.split_X_y(train_df, test_df)

    model = fn.create_mlp_model(X_train = X_train)
    model = fn.train_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model = model)

    y_pred_descaled = fn.prediction_unseen(model, scaler, X_test)

if __name__ == "__main__":
    execute_pipeline(data_pipeline)

