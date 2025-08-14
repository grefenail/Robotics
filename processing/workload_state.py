""" Module to manage the current workload state. """

import json
import pandas as pd
import duckdb


class WorkloadState:
    """Class to store and manage the current workload state."""

    def __init__(self, db_path: str = "workload_state.duckdb"):
        # Dictionary of user_id -> user metrics
        self.users = {}
        # Dictionary for overall (global) metrics
        self.overall = {}
        self._cached_state = None
        self._state_dirty = True

        self.db_path = db_path
        self.last_backup_timestamp = None

    @property
    def state(self) -> pd.DataFrame:
        """
        Return the full state, including user-level and overall metrics.
        This makes it easy to pass a single object around if needed.
        """
        # print("before constructing state", self.overall)
        # print("before constructing state2", self.users)
        if self._state_dirty or self._cached_state is None:
            # wrap the dictionaries in a list so that the DataFrame has one row
            # self._cached_state = pd.DataFrame(
            #     {"users": self.users, "overall": self.overall}
            # )
            self._cached_state = pd.DataFrame(
                [{**self.overall, "users": self.users}]
            )
            self._state_dirty = False
        return self._cached_state

    def update_state(self, row: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point to update all metrics based on an incoming row.
        Returns the entire state after updating.
        """
        user_id = row.get("user_id")
        if pd.isna(user_id):
            print("Skipping row due to missing user_id.")
            return self.state

        # Initialize user-specific metrics if new
        if user_id not in self.users:
            self._init_user_metrics(user_id)

        # Update the user's raw counters/metrics
        self._update_user_metrics(user_id, row)

        # Compute user-derived metrics (averages, ratios, etc.)
        self._update_user_derived_metrics(user_id, row)

        # Finally, update the overall (global) averages across all users
        self._update_overall_averages()

        self._state_dirty = True
        return self.state

    def _init_user_metrics(self, user_id: str) -> None:
        """Initialize the metrics dictionary for a new user."""
        self.users[user_id] = {
            "query_count": 0,
            "total_execution_time": 0,
            "scanned": 0,
            "spilled": 0,
            "avg_spill": 0,
            "avg_execution_time": 0,
            "queue_time_percentage": 0,
            "compile_overhead_ratio": 0,
            "query_type_counts": {},
            "total_joins": 0,
            "total_aggregations": 0,
            "unique_tables": set(),
            "cluster_metrics": {},
            "aborted_queries": 0,
            "abort_rate": 0,
            "read_write_ratio": 0,
            "timestamp": pd.Timestamp.min,
            "serverless": False,
        }

    def _update_user_metrics(self, user_id: str, row: pd.DataFrame) -> None:
        """Update raw counters (no derived calculations) for a user."""
        user_data = self.users[user_id]

        # Basic increments
        user_data["query_count"] += 1
        user_data["total_execution_time"] += (
            row.get("compile_duration_ms", 0)
            + row.get("queue_duration_ms", 0)
            + row.get("execution_duration_ms", 0)
        )
        user_data["scanned"] += row.get("mbytes_scanned", 0)
        user_data["spilled"] += row.get("mbytes_spilled", 0)

        # Query types
        query_type = row.get("query_type", "unknown")
        user_data["query_type_counts"][query_type] = (
            user_data["query_type_counts"].get(query_type, 0) + 1
        )

        # Joins, aggregations
        user_data["total_joins"] += row.get("num_joins", 0)
        user_data["total_aggregations"] += row.get("num_aggregations", 0)

        # Unique tables
        table_ids = row.get("read_table_ids", []) or []
        user_data["unique_tables"].update(table_ids)

        # Cluster metrics
        cluster_size = row.get("cluster_size", "unknown")
        if pd.isna(cluster_size):
            cluster_size = "unknown"

        if cluster_size not in user_data["cluster_metrics"]:
            user_data["cluster_metrics"][cluster_size] = {
                "query_count": 0,
                "total_duration": 0,
            }
        user_data["cluster_metrics"][cluster_size]["query_count"] += 1
        user_data["cluster_metrics"][cluster_size]["total_duration"] += row.get(
            "execution_duration_ms", 0
        )

        # is serverless if size of cluster is 0 or undefined
        user_data["serverless"] = row.get("cluster_size", 0) <= 0

        user_data["timestamp"] = max(
            row.get("arrival_timestamp", pd.Timestamp.min),
            user_data["timestamp"],
        )

        # Aborted queries
        if row.get("was_aborted", False):
            user_data["aborted_queries"] += 1

    def _update_user_derived_metrics(
        self, user_id: str, row: pd.DataFrame
    ) -> None:
        """Compute any user-level averages, ratios, or percentages."""
        user_data = self.users[user_id]
        qcount = user_data["query_count"]

        # Averages
        if qcount:
            user_data["avg_spill"] = round(user_data["spilled"] / qcount, 2)
            user_data["avg_execution_time"] = round(
                user_data["total_execution_time"] / qcount, 2
            )
        else:
            user_data["avg_spill"] = 0
            user_data["avg_execution_time"] = 0

        # Execution Efficiency
        total_execution_time = user_data["total_execution_time"]
        queue_duration = row.get("queue_duration_ms", 0)
        if total_execution_time <= 0:
            user_data["queue_time_percentage"] = 0
        else:
            user_data["queue_time_percentage"] = round(
                (queue_duration / total_execution_time) * 100, 2
            )

        compile_duration = row.get("compile_duration_ms", 0)
        execution_duration = row.get("execution_duration_ms", 0)
        if execution_duration <= 0:
            user_data["compile_overhead_ratio"] = 0
        else:
            user_data["compile_overhead_ratio"] = round(
                compile_duration / execution_duration, 2
            )

        # Aborted rate
        if qcount:
            user_data["abort_rate"] = round(
                (user_data["aborted_queries"] / qcount) * 100, 2
            )
        else:
            user_data["abort_rate"] = 0

        # Read/Write ratio
        read_ops = len(row.get("read_table_ids", []) or [])
        write_ops = len(row.get("write_table_ids", []) or [])
        # If there are zero write_ops, decide how to handle
        if write_ops == 0:
            # If we want to reflect infinite or undefined, we could change.
            # Here, we'll just do float('inf') if read_ops > 0, else 0.
            if read_ops > 0:
                user_data["read_write_ratio"] = float("inf")
            else:
                user_data["read_write_ratio"] = 0
        else:
            user_data["read_write_ratio"] = round(read_ops / write_ops, 2)

    def _update_overall_averages(self) -> None:
        """Compute global averages across all users and store them."""
        total_users = len(self.users)
        if total_users == 0:
            self.overall = {}
            return

        # Accumulators
        total_query_count = 0
        total_exec_time = 0
        total_scanned = 0
        total_spilled = 0
        total_abort_rate = 0
        timestamp = pd.Timestamp.min

        # Sum over all users
        for user_data in self.users.values():
            total_query_count += user_data["query_count"]
            total_exec_time += user_data["total_execution_time"]
            total_scanned += user_data["scanned"]
            total_spilled += user_data["spilled"]
            total_abort_rate += user_data["abort_rate"]
            timestamp = max(user_data["timestamp"], timestamp)

        # Compute averages
        self.overall["avg_query_count"] = round(
            total_query_count / total_users, 2
        )
        self.overall["avg_execution_time"] = round(
            total_exec_time / total_users, 2
        )
        self.overall["avg_scanned"] = round(total_scanned / total_users, 2)
        self.overall["avg_spilled"] = round(total_spilled / total_users, 2)
        self.overall["avg_abort_rate"] = round(
            total_abort_rate / total_users, 2
        )
        self.overall["timestamp"] = timestamp
        self.overall["total_queries"] = total_query_count
        self.overall["total_exec_time"] = total_exec_time

        self.overall["predicted_query_count"] = []
        self.overall["alerts"] = []
        self.overall["predicted_spill"] = []

    def reset_state(self) -> None:
        """Reset all user data and overall metrics."""
        self.users = {}
        self.overall = {}

    async def save_state(self) -> None:
        """
        Append a snapshot of the current state to DuckDB.
        The backup is stored in a table 'state_backup' as JSON strings.
        """
        con = duckdb.connect(self.db_path)
        # Create the backup table if it doesn't exist.
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS state_backup (
                backup_time TIMESTAMP,
                users TEXT,
                overall TEXT
            )
        """
        )
        backup_time = pd.Timestamp.now()
        # Serialize the dictionaries. The lambda converts any sets to lists.
        users_json = json.dumps(
            self.users, default=lambda o: list(o) if isinstance(o, set) else o
        )
        overall_json = json.dumps(self.overall, default=str)
        con.execute(
            "INSERT INTO state_backup VALUES (?, ?, ?)",
            (backup_time, users_json, overall_json),
        )
        con.close()
        print(f"Backup state saved at {backup_time}.")

    def load_state(self) -> None:
        """
        Load the most recent backup from DuckDB into memory.
        Converts lists back into sets for the 'unique_tables' field.
        """
        con = duckdb.connect(self.db_path)
        try:
            df = con.execute(
                "SELECT * FROM state_backup ORDER BY backup_time DESC LIMIT 1"
            ).df()
            if not df.empty:
                backup_row = df.iloc[0]
                self.users = json.loads(backup_row["users"])
                self.overall = json.loads(backup_row["overall"])
                # Convert unique_tables back to sets.
                for uid, metrics in self.users.items():
                    if "unique_tables" in metrics and isinstance(
                        metrics["unique_tables"], list
                    ):
                        metrics["unique_tables"] = set(metrics["unique_tables"])
                print(
                    f"Loaded state from backup at {backup_row['backup_time']}."
                )
            else:
                print("No backup found. Starting with an empty state.")
                self.reset_state()
        except Exception as e:
            print("Error loading state backup:", e)
            self.reset_state()
        finally:
            con.close()
