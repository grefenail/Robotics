import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MovingAveragePredictor:
    def __init__(self, window_size=50, step_interval=10, threshold=20000):
        self.window_size = window_size
        self.step_interval = step_interval
        self.threshold = threshold
        self.historical_data = []
        self.subpredictions = []
        self.initialized = False

    def moving_average(self, series, window):
        return (
            np.mean(series[-window:])
            if len(series) >= window
            else np.mean(series)
        )

    def initialize_model(self):
        if len(self.historical_data) >= self.window_size:
            self.initialized = True

    def predict_future(self, steps=10):
        if not self.initialized:
            return None
        last_avg = self.moving_average(self.historical_data, self.window_size)
        return [last_avg] * steps

    def visualize_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(
            range(len(self.historical_data)),
            self.historical_data,
            label="Historical Data",
        )
        for i, subprediction in enumerate(self.subpredictions):
            start_idx = len(self.historical_data) + (i * self.step_interval)
            plt.plot(
                range(start_idx, start_idx + len(subprediction)),
                subprediction,
                linestyle="dashed",
                label=f"Prediction {i+1}",
            )
        plt.xlabel("Time Steps")
        plt.ylabel("Spilled Data Volume")
        plt.legend()
        plt.title("Real-Time Spill Prediction Using Moving Average")
        plt.show()
        print("Prediction process completed. Visualization generated.")

    def predict(self, new_data) -> tuple[(list | None), str]:
        self.historical_data.extend(new_data)
        if (
            not self.initialized
            and len(self.historical_data) >= self.window_size
        ):
            self.initialize_model()

        if self.initialized:
            future_pred = self.predict_future(steps=self.step_interval)
            alert = ""
            self.subpredictions.append(future_pred)
            # self.visualize_results()

            if any(pred >= self.threshold for pred in future_pred):
                alert = f"Warning: Predicted spill volume exceeds threshold of {self.threshold} MB!"
                print(alert)

            return future_pred, alert
        return None, None


class RealTimePredictor:
    def __init__(self, window_size=200, step_interval=50, threshold=80):
        self.predictor = MovingAveragePredictor(
            window_size=window_size,
            step_interval=step_interval,
            threshold=threshold,
        )
        self.previous_data = []
        self.step_size = step_interval

    def predict_rt_data(self, df: pd.DataFrame):
        spill_amount = df["avg_spilled"].iloc[0]
        self.previous_data.append(spill_amount)
        if len(self.previous_data) > self.step_size:
            prediction, alert = self.predictor.predict(self.previous_data)
            df["predicted_spill"] = [prediction]
            if alert:
                df.at[0, "alerts"].append(alert)
            self.previous_data = []
        # if "predicted_spill" in df:
        #     print(df["predicted_spill"].iloc[0])
        return df


if __name__ == "__main__":
    real_spill_data = [
        pd.DataFrame({"avg_spilled": spilled})
        for spilled in np.random.randint(10, 100, 100)
    ]
    rt_predictor = RealTimePredictor(
        window_size=200, step_interval=50, threshold=80
    )
    for tmp_df in real_spill_data:
        rt_predictor.predict_rt_data(tmp_df)
