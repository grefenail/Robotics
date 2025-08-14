import pandas as pd


class BillingCalculator:
    """Calculates billing per user and stores data in memory for tracking over time."""

    def __init__(self):
        # Pricing models
        # $0.0005 per execution second
        self.serverless_price_per_second = 0.0005
        
        # $0.0002 per MB scanned
        self.serverless_price_per_mb_scanned = 0.0002
        # $0.0003 per MB spilled
        self.serverless_price_per_mb_spilled = 0.0003

        # $0.0003 per execution second
        self.provisioned_price_per_second = 0.0003

        # $0.00015 per MB scanned
        self.provisioned_price_per_mb_scanned = 0.00015
        # $0.00025 per MB spilled
        self.provisioned_price_per_mb_spilled = 0.00025
        self.provisioned_cluster_size_price_per_second = 0.15 / 3600

        # In-memory storage
        self.billing_data = {}

    def calculate_user_billing(self, user_id, user_data):
        """Calculates billing for a single user and tracks accumulated costs over time."""
        execution_time = (
            user_data.get("total_execution_time", 0) / 1000
        )  # Convert ms to seconds
        scanned_data = user_data.get("scanned", 0)  # MB scanned
        spilled_data = user_data.get("spilled", 0)  # MB spilled
        is_serverless = user_data.get("serverless", False)  # Identify workload type
        cluster_size = user_data.get("cluster_size", 0)
        cluster_cost = 0

        # Apply different billing models based on workload type
        if is_serverless:
            execution_cost = execution_time * self.serverless_price_per_second
            scanned_cost = scanned_data * self.serverless_price_per_mb_scanned
            spilled_cost = spilled_data * self.serverless_price_per_mb_spilled
        else:
            execution_cost = execution_time * self.provisioned_price_per_second
            scanned_cost = scanned_data * self.provisioned_price_per_mb_scanned
            spilled_cost = spilled_data * self.provisioned_price_per_mb_spilled
            cluster_cost = (
                cluster_size * self.provisioned_cluster_size_price_per_second
            )

        total_cost = execution_cost + scanned_cost + spilled_cost + cluster_cost

        # Store per-user billing information
        user_data["billing"] = {
            "execution_cost": round(execution_cost, 4),
            "scanned_cost": round(scanned_cost, 4),
            "spilled_cost": round(spilled_cost, 4),
            "total_cost": round(total_cost, 4),
            "serverless": is_serverless,
        }

        return total_cost

    def calculate_all_users_billing(self, state):
        """Calculates billing for all users in the workload state and updates in-memory storage."""

        # Check if state is a Pandas DataFrame and extract dictionary values
        if isinstance(state, pd.DataFrame) and not state.empty:
            state_dict = state.iloc[0].to_dict()  # Convert first row to a dictionary
            users = state_dict.get("users", {})
        else:
            users = state.get("users", {})
            state_dict = state

        if not users:
            print("No user data available for billing calculation.")
            return state

        total_billing = 0

        for user_id, user_data in users.items():
            user_total_cost = self.calculate_user_billing(user_id, user_data)
            total_billing += user_total_cost
            users[user_id]["total_cost"] = round(user_total_cost, 4)  # Store total cost per user

            # Update billing data in memory
            if user_id in self.billing_data:
                self.billing_data[user_id]["execution_cost"] += user_data["billing"]["execution_cost"]
                self.billing_data[user_id]["scanned_cost"] += user_data["billing"]["scanned_cost"]
                self.billing_data[user_id]["spilled_cost"] += user_data["billing"]["spilled_cost"]
                self.billing_data[user_id]["total_cost"] += user_data["billing"]["total_cost"]
            else:
                self.billing_data[user_id] = {
                    "execution_cost": user_data["billing"]["execution_cost"],
                    "scanned_cost": user_data["billing"]["scanned_cost"],
                    "spilled_cost": user_data["billing"]["spilled_cost"],
                    "total_cost": user_data["billing"]["total_cost"],
                    "serverless": user_data["billing"]["serverless"],
                }

        # Store overall billing in extracted dictionary
        state_dict["total_billing"] = round(total_billing, 4)

        # Reconstruct the state DataFrame with updated values
        updated_state = pd.DataFrame([{**state_dict, "users": users}])

        # print(f"✅ Billing calculated. Total system cost: ${total_billing:.4f}")
        return updated_state  # Return updated DataFrame
