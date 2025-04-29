import dask.dataframe as dd
from dask.distributed import Client
import xgboost as xgb
import pandas as pd
import os
import logging
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Dask-trained XGBoost model to a new network")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to trained XGBoost .json Booster model")
    parser.add_argument("--target", type=str, required=True, help="Path to .parquet file for inference")
    parser.add_argument("--save_name", type=str, required=True, help="Filename for output")
    return parser.parse_args()

def read_inferred_network_dask(inferred_network_file):
    df = dd.read_parquet(inferred_network_file)
    df["source_id"] = df["source_id"].str.upper()
    df["target_id"] = df["target_id"].str.upper()
    return df

def main():
    args = parse_args()

    model_path = args.model
    target_path = args.target
    output_dir = args.output_dir
    save_name = args.save_name

    logging.info("Connecting to Dask client")
    client = Client(processes=False)

    logging.info("Loading XGBoost Booster")
    booster = xgb.Booster()
    booster.load_model(model_path)

    feature_names_str = booster.attr("feature_names")
    if feature_names_str is None:
        raise ValueError("Missing feature_names in model attributes. Use booster.set_attr(...) during training.")
    feature_names = feature_names_str.split(",")

    logging.info("Reading inferred network")
    inferred_dd = read_inferred_network_dask(target_path)
    X_dd = inferred_dd[feature_names]

    logging.info("Converting to DaskDMatrix")
    dtest = xgb.dask.DaskDMatrix(client, X_dd, feature_names=feature_names)

    logging.info("Running distributed prediction")
    y_pred = xgb.dask.predict(client, booster, dtest)

    # Convert to pandas (merging Dask DataFrame + Dask array)
    logging.info("Joining predictions back to source-target pairs")
    result_df = inferred_dd[["source_id", "target_id"]].compute()
    result_df["score"] = y_pred.compute()
    result_df = result_df.drop_duplicates()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, save_name)
    logging.info(f"Saving to {output_path}")
    result_df.to_csv(output_path, sep="\t", index=False)

    client.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
