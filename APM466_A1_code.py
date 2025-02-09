import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import re

class BondDataProcessor:
    def __init__(self, file_path):
        self.bonds_data_stored = pd.read_excel(file_path, sheet_name="Bond Data")
        self.bonds_data = self.bonds_data_stored.copy()
        self.process_bond_data()

    def process_bond_data(self):
        for i, row in self.bonds_data_stored.iterrows():
            for col in self.bonds_data_stored.columns[2:]:
                day = float(col.strftime('%d'))
                coupon = float(row[0].split()[1])

                try:
                    price = float(row[col])
                except ValueError:
                    continue
                self.bonds_data.at[i, col] = price + ((4 * 30 + (day - 1)) / 360) * coupon

    def get_bonds_data(self):
        return self.bonds_data


class YieldCalculator:
    def __init__(self, bonds_data):
        self.yield_stored = bonds_data.copy()
        self.calculate_yields()

    def calculate_yields(self):
        for i, row in self.yield_stored.iterrows():
            for col in self.yield_stored.columns[2:]:
                coupon = float(row[0].split()[1])
                maturity = int(row[1])
                semi_annual_coupon_f = 2

                try:
                    price = float(row[col])
                except ValueError:
                    continue

                total_periods = int(maturity * semi_annual_coupon_f)
                fractional_period = (maturity * semi_annual_coupon_f) - total_periods

                cash_flow = []
                if total_periods > 0:
                    cash_flow = [coupon / semi_annual_coupon_f] * total_periods
                    cash_flow[-1] += 100
                else:
                    final_coupon = (coupon / semi_annual_coupon_f) * fractional_period
                    cash_flow = [final_coupon + 100]

                cash_flow.insert(0, -price)

                if cash_flow:
                    irr = npf.irr(cash_flow)
                    irr_pct = irr * 100
                else:
                    print("Error: Cash flows list is empty, cannot compute IRR.")
                    continue

                self.yield_stored.at[i, col] = irr_pct * 2

        self.yield_stored["Maturity Date"] = self.yield_stored["Date"].apply(lambda x: " ".join(re.findall(r"([A-Za-z]+ \d{2})", x)))
        self.yield_stored["Maturity Date"] = pd.to_datetime(self.yield_stored["Maturity Date"], format="%b %y")
        self.yield_stored = self.yield_stored.drop(columns=["Maturity"])
        self.yield_stored = self.yield_stored.sort_values(by="Maturity Date")

    def get_yield_data(self):
        return self.yield_stored

    def plot_yields(self):
        plt.figure(figsize=(12, 6))
        maturity_dates = []
        average_yields = []
        min_yields = []
        max_yields = []

        for index, row in self.yield_stored.iterrows():
            x_value = row["Maturity Date"]
            y_values = row.drop(["Maturity Date", "Date"]).values.astype(float)
            plt.scatter([x_value] * len(y_values), y_values, label=row["Date"])

            avg_yield = np.mean(y_values)
            min_yield = np.min(y_values)
            max_yield = np.max(y_values)

            maturity_dates.append(x_value)
            average_yields.append(avg_yield)
            min_yields.append(min_yield)
            max_yields.append(max_yield)

        plt.plot(maturity_dates, average_yields, color="black", linestyle="-", linewidth=2, label="Average Yield")
        plt.fill_between(maturity_dates, min_yields, max_yields, color="gray", alpha=0.2, label="Min-Max Range")
        plt.xlabel("Maturity Date")
        plt.ylabel("Yield")
        plt.title("Bond Yields Over Maturity")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig("yields_plot.png", dpi=300, bbox_inches="tight")
        plt.show()


class SpotRateCalculator:
    def __init__(self, bonds_data):
        self.spot_rate_stored = bonds_data.copy()
        self.calculate_spot_rates()

    def calculate_spot_rates(self):
        coupon = float(self.spot_rate_stored.iloc[0, 0].split()[1].strip())
        for col in self.spot_rate_stored.columns[2:]:
            self.spot_rate_stored.at[0, col] = -1 / self.spot_rate_stored.iloc[0, 1] * np.log(self.spot_rate_stored.at[0, col] / (100 + coupon / 2))

        for row_index in range(1, len(self.spot_rate_stored)):
            coupon = float(self.spot_rate_stored.iloc[row_index, 0].split()[1].strip())
            for col in self.spot_rate_stored.columns[2:]:
                exp_term = 0
                for prev_row in range(row_index):
                    exp_term += (coupon / 2) * np.exp(-1 * self.spot_rate_stored.at[prev_row, col] * self.spot_rate_stored.iloc[prev_row, 1])
                self.spot_rate_stored.at[row_index, col] = -1 / self.spot_rate_stored.iloc[row_index, 1] * np.log(
                    (self.spot_rate_stored.at[row_index, col] - exp_term) / (100 + coupon / 2)
                )

        self.spot_rate_stored["Maturity Date"] = self.spot_rate_stored["Date"].apply(lambda x: " ".join(re.findall(r"([A-Za-z]+ \d{2})", x)))
        self.spot_rate_stored["Maturity Date"] = pd.to_datetime(self.spot_rate_stored["Maturity Date"], format="%b %y")
        self.spot_rate_stored = self.spot_rate_stored.sort_values(by="Maturity Date")

    def get_spot_rate_data(self):
        return self.spot_rate_stored

    def plot_spot_rates(self):
        plt.figure(figsize=(12, 6))
        maturity_dates = []
        average_yields = []
        min_yields = []
        max_yields = []

        for index, row in self.spot_rate_stored.iterrows():
            x_value = row["Maturity Date"]
            y_values = row.drop(["Maturity Date", "Date"]).values.astype(float)
            plt.scatter([x_value] * len(y_values), y_values, label=row["Date"])

            avg_yield = np.mean(y_values)
            min_yield = np.min(y_values)
            max_yield = np.max(y_values)

            maturity_dates.append(x_value)
            average_yields.append(avg_yield)
            min_yields.append(min_yield)
            max_yields.append(max_yield)

        plt.plot(maturity_dates, average_yields, color="black", linestyle="-", linewidth=2, label="Average Spot Rate")
        plt.fill_between(maturity_dates, min_yields, max_yields, color="gray", alpha=0.2, label="Min-Max Range")
        plt.xlabel("Maturity Date")
        plt.ylabel("Spot Rate")
        plt.title("Spot Rates Over Maturity")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig("spot_rates_plot.png", dpi=300, bbox_inches="tight")
        plt.show()


class ForwardRateCalculator:
    def __init__(self, spot_rate_data):
        self.forward_rate_stored = spot_rate_data[spot_rate_data.index % 2 == 0].copy()
        self.calculate_forward_rates()

    def calculate_forward_rates(self):
        for i, row in self.forward_rate_stored.iterrows():
            for col in self.forward_rate_stored.columns[1:-1]:
                if i != 8:
                    self.forward_rate_stored.at[i + 2, col] = ((i / 2 + 2) * self.forward_rate_stored.at[i + 2, col] - self.forward_rate_stored.at[0, col]) / (i / 2 + 1)
        self.forward_rate_stored = self.forward_rate_stored.drop(self.forward_rate_stored.index[0]).reset_index(drop=True)

    def get_forward_rate_data(self):
        return self.forward_rate_stored

    def plot_forward_rates(self):
        plt.figure(figsize=(12, 6))
        maturity_dates = []
        average_yields = []
        min_yields = []
        max_yields = []

        for index, row in self.forward_rate_stored.iterrows():
            x_value = row["Maturity Date"]
            y_values = row.drop(["Maturity Date", "Date"]).values.astype(float)
            plt.scatter([x_value] * len(y_values), y_values, label=row["Date"])

            avg_yield = np.mean(y_values)
            min_yield = np.min(y_values)
            max_yield = np.max(y_values)

            maturity_dates.append(x_value)
            average_yields.append(avg_yield)
            min_yields.append(min_yield)
            max_yields.append(max_yield)

        plt.plot(maturity_dates, average_yields, color="black", linestyle="-", linewidth=2, label="Average Forward Rate")
        plt.fill_between(maturity_dates, min_yields, max_yields, color="gray", alpha=0.2, label="Min-Max Range")
        plt.xlabel("Maturity Date")
        plt.ylabel("Forward Rate")
        plt.title("Forward Rates Over Maturity")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig("forward_rates_plot.png", dpi=300, bbox_inches="tight")
        plt.show()


class CovarianceMatrixCalculator:
    def __init__(self, yield_df, forward_rate_df):
        self.yield_stored = yield_df.drop(columns=["Date", "Maturity Date"]).values
        self.forward_data = forward_rate_df.drop(columns=["Date", "Maturity Date"]).values
        self.cov_matrix_yield, self.cov_matrix_forward = self.calculate_covariance_matrices()

    def calculate_covariance_matrices(self):
        log_returns_yield = np.log(self.yield_stored[:, 1:] / self.yield_stored[:, :-1])
        cov_matrix_yield = np.cov(log_returns_yield)

        log_returns_forward = np.log(self.forward_data[:, 1:] / self.forward_data[:, :-1])
        cov_matrix_forward = np.cov(log_returns_forward)

        cov_df_yield = pd.DataFrame(cov_matrix_yield, 
                                    index=self.yield_stored.index, 
                                    columns=self.yield_stored.index)

        cov_df_forward = pd.DataFrame(cov_matrix_forward, 
                                    index=self.forward_data.index, 
                                    columns=self.forward_data.index)

        return cov_df_yield, cov_df_forward

    def get_covariance_matrices(self):
        return self.cov_matrix_yield, self.cov_matrix_forward

    def print_eigenvalues_and_eigenvectors(self):
        eigenval_yield, eigenvec_yield = np.linalg.eig(self.cov_matrix_yield.values)
        eigenval_forward, eigenvec_forward = np.linalg.eig(self.cov_matrix_forward.values)

        print("Eigenvalues of Yield Covariance Matrix:")
        print(eigenval_yield)

        print("\nEigenvectors of Yield Covariance Matrix:")
        print(eigenvec_yield)

        print("\nEigenvalues of Forward Rate Covariance Matrix:")
        print(eigenval_forward)

        print("\nEigenvectors of Forward Rate Covariance Matrix:")
        print(eigenvec_forward)


# Main execution
if __name__ == "__main__":
    file_path = "APM466_A1_Data.xlsx"
    
    bond_processor = BondDataProcessor(file_path)
    bonds_data = bond_processor.get_bonds_data()

    yield_calculator = YieldCalculator(bonds_data)
    yield_data = yield_calculator.get_yield_data()
    #yield_calculator.plot_yields()

    spot_rate_calculator = SpotRateCalculator(bonds_data)
    spot_rate_data = spot_rate_calculator.get_spot_rate_data()
    #spot_rate_calculator.plot_spot_rates()

    forward_rate_calculator = ForwardRateCalculator(spot_rate_data)
    forward_rate_data = forward_rate_calculator.get_forward_rate_data()
    #forward_rate_calculator.plot_forward_rates()

    cov_matrix_calculator = CovarianceMatrixCalculator(yield_data, forward_rate_data)
    #cov_matrix_calculator.print_eigenvalues_and_eigenvectors()