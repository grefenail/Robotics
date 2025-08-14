import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ARIMAPredictor:
    def __init__(
        self,
        window_size=50,
        step_interval=10,
        max_d=2,
        p_range=(0, 3),
        q_range=(0, 3),
    ):
        """
        Initialize the ARIMAPredictor.

        Parameters:
        - window_size: Number of data points to use for training.
        - step_interval: Number of steps to predict each time.
        - max_d: Maximum order of differencing to achieve stationarity.
        - p_range: Tuple indicating the range of p values to consider.
        - q_range: Tuple indicating the range of q values to consider.
        - select_params: Whether to perform parameter selection.
        """
        self.window_size = window_size
        self.step_interval = step_interval
        self.max_d = max_d
        self.p_range = p_range
        self.q_range = q_range
        self.historical_data = []
        self.subpredictions = []
        self.model_fit = None
        self.arima_order = None
        self.initialized = False

    @staticmethod
    def check_stationarity(timeseries):
        """
        Check the stationarity of a time series using the Augmented Dickey-Fuller test.

        Returns:
        - True if the series is stationary, False otherwise.
        """
        result = adfuller(timeseries)
        return result[1] < 0.05  # If p-value < 0.05, data is stationary

    @staticmethod
    def difference_series(series, order=1):
        """
        Difference a time series to achieve stationarity.

        Parameters:
        - series: The original time series.
        - order: The order of differencing.

        Returns:
        - Differenced time series.
        """
        return np.diff(series, n=order)

    def find_stationary_d(self, series):
        """
        Determine the degree of differencing required to make the series stationary.

        Parameters:
        - series: The time series data.

        Returns:
        - d: The determined order of differencing.
        - differenced_series: The differenced time series.
        """
        d = 0
        temp_series = series.copy()
        while not self.check_stationarity(temp_series) and d < self.max_d:
            temp_series = self.difference_series(temp_series, order=1)
            d += 1
            print(f"Differenced series to order {d} for stationarity check.")
        return d, temp_series

    def select_arima_order(self, series):
        """
        Select the best ARIMA(p, d, q) order based on AIC.

        Parameters:
        - series: The time series data.

        Returns:
        - best_order: Tuple of (p, d, q) with the lowest AIC.
        """
        # Determine d
        d, _ = self.find_stationary_d(series)
        if d > self.max_d:
            print(
                "Maximum differencing order reached. Series may still be non-stationary."
            )

        # Define p and q ranges
        p_values = range(self.p_range[0], self.p_range[1] + 1)
        q_values = range(self.q_range[0], self.q_range[1] + 1)
        best_aic = np.inf
        best_order = None

        print(f"Selecting ARIMA parameters with d={d}...")
        for p, q in itertools.product(p_values, q_values):
            try:
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                print(f"Tested ARIMA({p},{d},{q}) - AIC:{model_fit.aic}")
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
            except Exception as e:
                print(f"ARIMA({p},{d},{q}) failed to fit. Error: {e}")
                continue

        if best_order is None:
            # Fallback to (0, d, 0) if no model could be fitted
            best_order = (0, d, 0)
            print(
                f"No suitable ARIMA model found. Falling back to ARIMA{best_order}."
            )
        else:
            print(f"Selected ARIMA{best_order} with AIC={best_aic}")

        return best_order

    def initialize_model(self):
        """
        Initialize the ARIMA model by selecting parameters and fitting the model.
        """
        if len(self.historical_data) >= self.window_size:
            train_data = self.historical_data[-self.window_size :]
            order = self.select_arima_order(train_data)

            self.arima_order = order
            self.model_fit = self.train_arima(train_data, order)
            self.initialized = True
            print(f"Initialized ARIMA model with order {order}")

    def train_arima(self, series, order):
        """
        Train the ARIMA model.

        Parameters:
        - series: The training time series data.
        - order: Tuple of (p, d, q).

        Returns:
        - Fitted ARIMA model.
        """
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            return model_fit
        except Exception as e:
            print(f"Failed to train ARIMA{order}. Error: {e}")
            return None

    @staticmethod
    def predict_future(model_fit, steps=10):
        """
        Predict future values using the fitted ARIMA model.

        Parameters:
        - model_fit: The fitted ARIMA model.
        - steps: Number of future steps to predict.

        Returns:
        - Forecasted values as a NumPy array.
        """
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def visualize_results_with_subpredictions(self):
        """
        Visualize the historical data along with sub-predictions.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            range(len(self.historical_data)),
            self.historical_data,
            label="Historical Data",
        )
        for i, subprediction in enumerate(self.subpredictions):
            start_idx = (
                self.step_interval * i
                + len(self.historical_data)
                - len(self.subpredictions) * self.step_interval
            )
            plt.plot(
                range(start_idx, start_idx + len(subprediction)),
                subprediction,
                linestyle="dashed",
            )
        plt.xlabel("Time Steps")
        plt.ylabel("Query Counts")
        plt.legend()
        plt.title("Real-Time Workload Prediction Using ARIMA")
        plt.show()

    def predict(self, new_data):
        """
        Input new data into the predictor and perform prediction if enough data is available.

        Parameters:
        - new_data: A list or array of new data points to be added.

        Returns:
        - Forecasted values if prediction is made, else None.
        """
        self.historical_data.extend(new_data)
        # print(f"Added new data: {new_data}")

        if (
            not self.initialized
            and len(self.historical_data) >= self.window_size
        ):
            self.initialize_model()

        if self.initialized:
            try:
                # Retrain the model with new data without parameter selection
                train_data = self.historical_data[-self.window_size :]
                model_fit = self.train_arima(train_data, self.arima_order)
                if model_fit is not None:
                    # Predict future steps
                    future_pred = self.predict_future(
                        model_fit, steps=self.step_interval
                    )
                    self.subpredictions.append(future_pred)
                    # print(
                    #     f"Forecasted next {self.step_interval} steps: {future_pred}"
                    # )
                    return future_pred
                else:
                    print("Model fitting failed.")
                    return None
            except Exception as e:
                print(f"Prediction failed. Error: {e}")
                return None
        else:
            # print("Not enough data to initialize the model yet.")
            return None


class RealTimePredictor:
    """Class to handle real-time data streaming and predictions."""

    def __init__(self, window_size=100, step_interval=10):
        self.predictor = ARIMAPredictor(
            window_size=window_size, step_interval=step_interval
        )
        self.previous_data = []
        self.step_size = step_interval

    def predict_rt_data(self, df: pd.DataFrame):
        """Predict real-time data using ARIMA."""
        query_count = df["avg_query_count"].iloc[0]
        self.previous_data.append(query_count)
        if len(self.previous_data) > self.step_size:
            df["predicted_query_count"] = [
                self.predictor.predict(self.previous_data)
            ]
            self.previous_data = []  # Reset for next batch
        # if predicted_query_count is in df overall, print
        # if "predicted_query_count" in df:
        #     print(df["predicted_query_count"])
        return df

    def visualize_predictions(self):
        """Visualize the predictions."""
        self.predictor.visualize_results_with_subpredictions()


if __name__ == "__main__":
    # Simulate real-time data
    np.random.seed(42)
    x = np.linspace(0, 20, 3000)
    query_counts = (
        10 * x**3
        - 5 * x**2
        + 3 * x
        + 50
        + 50 * np.sin(2 * np.pi * x / 5)
        + np.random.normal(0, 20, len(x))
    )
    query_counts = query_counts.astype(int).tolist()

    query_counts_df = [
        pd.DataFrame({"avg_query_count": query_count})
        for query_count in query_counts
    ]

    # Instantiate the real-time predictor
    rt_predictor = RealTimePredictor(window_size=200, step_interval=50)

    # Simulate real-time data feeding
    for tmp_df in query_counts_df:
        rt_predictor.predict_rt_data(tmp_df)

    # Visualize the results
    rt_predictor.visualize_predictions()
