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

def read_inferred_network(inferred_network_file: str) -> dd.DataFrame:
    """
    Loads a melted sparse inferred network from Parquet and pivots it into a Dask DataFrame
    where each row is (source_id, target_id) and columns are score_types (mean-aggregated).
    """
    logging.info(f"Loading melted sparse network from: {inferred_network_file}")
    melted_ddf = dd.read_parquet(inferred_network_file, engine="pyarrow")

    # Standardize IDs
    melted_ddf["source_id"] = melted_ddf["source_id"].str.upper()
    melted_ddf["target_id"] = melted_ddf["target_id"].str.upper()

    # Aggregate scores
    grouped_ddf = (
        melted_ddf
        .groupby(["source_id", "peak_id", "target_id", "score_type"])["score_value"]
        .mean()
        .reset_index()
    )

    # Pivot manually by converting to pandas (if dataset is small enough)
    def pivot_partition(df):
        return df.pivot_table(
            index=["source_id", "peak_id", "target_id"],
            columns="score_type",
            values="score_value",
            aggfunc="first"
        ).reset_index()

    # Apply pivot in a single partition (best if you've already aggregated)
    pivot_df = grouped_ddf.compute()  # convert to Pandas here
    pivot_df = pivot_partition(pivot_df)
    return dd.from_pandas(pivot_df, npartitions=1)

def main():
    args = parse_args()

    model_path = args.model
    target_path = args.target
    output_dir = args.output_dir
    save_name = args.save_name

    logging.info("Loading XGBoost Booster")
    booster = xgb.Booster()
    booster.load_model(model_path)

    logging.info("Reading inferred network")
    inferred_dd = read_inferred_network(target_path)
    
    feature_names = booster.feature_names
    
    X_dd = inferred_dd[feature_names]

    logging.info("Converting to DaskDMatrix")
    client = Client()
    dtest = xgb.dask.DaskDMatrix(data=X_dd, feature_names=feature_names, client=client)

    logging.info("Running distributed prediction")
    y_pred = xgb.dask.predict(client=client, model=booster, data=dtest)

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
    logging.info("Done!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
