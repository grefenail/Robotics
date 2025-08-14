"""
Load the cleaned data, process it and send to the dashboard.
"""

import asyncio
import pandas as pd
from processing.billing_processing import BillingCalculator
from processing.helpers import get_rows, load_data, upload_data
from processing.workload_state import WorkloadState
from processing.prediction import RealTimePredictor
from processing.prediction_spill import RealTimePredictor as SpillPredictor

STATE_STORAGE_TIMER = 120
rt_predictor = RealTimePredictor(window_size=200, step_interval=100)
spill_predictor = SpillPredictor(
    window_size=200, step_interval=100, threshold=15000
)
billing_calculator = BillingCalculator()


async def process_dataframe(
    df: pd.DataFrame, state: WorkloadState
) -> pd.DataFrame:
    """Process a DataFrame in a more efficient, batched fashion."""
    processed_states = []

    # itertuples usually faster than iterrows
    for row in df.itertuples(index=False, name="Row"):
        row_dict = row._asdict()

        updated_state = await asyncio.to_thread(state.update_state, row_dict)

        # Save state asynchronously.
        overall_ts = updated_state["timestamp"].iloc[0]
        if overall_ts is not None:
            # convert to a Python datetime if it's a pandas Timestamp
            overall_dt = (
                overall_ts.to_pydatetime()
                if isinstance(overall_ts, pd.Timestamp)
                else overall_ts
            )
            # if this is the first valid timestamp
            if state.last_backup_timestamp is None:
                print("initializing last_backup_timestamp")
                state.last_backup_timestamp = overall_dt
            # Calculate the elapsed time in seconds.
            elapsed = (overall_dt - state.last_backup_timestamp).total_seconds()
            if elapsed >= STATE_STORAGE_TIMER:
                print(f"start saving state at: {overall_ts}")
                asyncio.create_task(state.save_state())
                state.last_backup_timestamp = overall_ts

        # If the predictor is running on CPU
        updated_state = await asyncio.to_thread(
            rt_predictor.predict_rt_data, updated_state
        )

        updated_state = await asyncio.to_thread(
            spill_predictor.predict_rt_data, updated_state
        )

        # Calculate billing for all users
        updated_state = await asyncio.to_thread(
            billing_calculator.calculate_all_users_billing, updated_state
        )
        processed_states.append(updated_state)

        # Let the event loop handle other tasks.
        await asyncio.sleep(0)

    return pd.concat(processed_states, ignore_index=True)


if __name__ == "__main__":
    partial_data = load_data("data/serverless/serverless_full.parquet", n=500)

    workload_state = WorkloadState()
    row_generator = get_rows(partial_data)

    for row in row_generator:
        data = process_dataframe(row, workload_state)
