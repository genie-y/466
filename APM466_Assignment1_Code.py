import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import re

bonds_data_stored = pd.read_excel("APM466_A1_Data.xlsx",sheet_name="Bond Data")
bonds_data = bonds_data_stored.copy()

for i, row in bonds_data_stored.iterrows():
    for col in bonds_data_stored.columns[2:]:
        day = float(col.strftime('%d'))  
        coupon = float(row[0].split()[1]) 

        try:
            price = float(row[col])
        except ValueError:
            continue
        bonds_data.at[i, col] = price + ((4*30 + (day - 1)) / 360) * coupon
#print(bonds_data)


yield_stored = bonds_data.copy()
for i, row in bonds_data_stored.iterrows():
    for col in bonds_data_stored.columns[2:]: 
        
        coupon = float(row[0].split()[1]) 
        maturity = int(row[1])
        semi_annual_coupon_f = 2 
        try:
            price = float(row[col])
        except ValueError:

            continue
        
        # Calculate the number of periods (semi-annually compounding here)
        Total_periods = int(maturity * semi_annual_coupon_f) 
        fractional_period = (maturity * semi_annual_coupon_f) - Total_periods 

        #print(f"Number of periods: {num_periods}") 

        # Make Cash Flow equation/timeline
        Cash_flow = []

        if Total_periods > 0:
            Cash_flow = [coupon / semi_annual_coupon_f] * Total_periods 
            Cash_flow[-1] += 100 # Add FV at the end
        else:
            final_coupon = (coupon / semi_annual_coupon_f) * fractional_period 
            Cash_flow = [final_coupon + 100]  # The final cash flow includes the coupon and face value at maturity

        Cash_flow.insert(0, -price)
        #print(f"Cash flows: {cash_flows}")


        #calculate IRR
        if Cash_flow:
            irr = npf.irr(Cash_flow)

            # Convert to annual percentage rate (APR) for easier interpretation
            irr_pct = irr * 100

            # print(f"The IRR of the bond is: {irr_percentage:.2f}%")
        else:
            print("Error: Cash flows list is empty, cannot compute IRR.")
        
        yield_stored.at[i, col] = irr_pct * 2


# Maturity Date 
yield_stored["Maturity Date"] = yield_stored["Date"].apply(lambda x: " ".join(re.findall(r"([A-Za-z]+ \d{2})", x)))
yield_stored["Maturity Date"] = pd.to_datetime(yield_stored["Maturity Date"], format="%b %y")
yield_stored = yield_stored.drop(columns=["Maturity"])

# Sort by Maturity Date
yield_stored = yield_stored.sort_values(by="Maturity Date")

#Plot the data
plt.figure(figsize=(12, 6))
maturity_dates = []
average_yields = []
min_yields = []
max_yields = []

for index, row in yield_stored.iterrows():
    x_value = row["Maturity Date"]  # The bond's maturity date on x-axis
    y_values = row.drop(["Maturity Date", "Date"]).values.astype(float)  # The bond's yields on y-axis
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
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside

# Display plot
plt.tight_layout()  # Adjust layout
plt.savefig("yields_plot.png", dpi=300, bbox_inches="tight")  # Save plot to PNG
#plt.show()
#print(yield_data)
cov_yield_data = yield_stored[yield_stored.index % 2 == 0]
#print(cov_yield_data)

spot_rate_stored = bonds_data.copy()
coupon = float(spot_rate_stored.iloc[0, 0].split()[1].strip())  # Extract the coupon percentage
for col in spot_rate_stored.columns[2:]:  # Skip the first 2 columns
    spot_rate_stored.at[0, col] = -1/spot_rate_stored.iloc[0, 1] * np.log(spot_rate_stored.at[0, col] / (100 + coupon / 2))

def calculate_spot_rate(spot_rate_data):
    for row_index in range(1, len(spot_rate_data)):
        coupon = float(spot_rate_data.iloc[row_index, 0].split()[1].strip())
        
        for col in spot_rate_data.columns[2:]:
            exp_term = 0
            # Loop through all previous rows to accumulate the exponent terms
            for prev_row in range(row_index):
                exp_term += (coupon / 2) * np.exp(-1 * spot_rate_data.at[prev_row, col] * spot_rate_data.iloc[prev_row, 1])
            
            spot_rate_data.at[row_index, col] = -1 / spot_rate_data.iloc[row_index, 1] * np.log(
                (spot_rate_data.at[row_index, col] - exp_term) / (100 + coupon / 2)
            )
    spot_rate_data["Maturity Date"] = spot_rate_data["Date"].apply(lambda x: " ".join(re.findall(r"([A-Za-z]+ \d{2})", x)))
    spot_rate_data["Maturity Date"] = pd.to_datetime(spot_rate_data["Maturity Date"], format="%b %y")
    spot_rate_data = spot_rate_data.sort_values(by="Maturity Date")
    return spot_rate_data

spot_rate_stored = calculate_spot_rate(spot_rate_stored)
#print(spot_rate_data)



spot_rate_stored["Maturity Date"] = spot_rate_stored["Date"].apply(lambda x: " ".join(re.findall(r"([A-Za-z]+ \d{2})", x)))
spot_rate_stored["Maturity Date"] = pd.to_datetime(spot_rate_stored["Maturity Date"], format="%b %y")
spot_rate_stored = spot_rate_stored.drop(columns=["Maturity"])
spot_rate_stored = spot_rate_stored.sort_values(by="Maturity Date")

# Plot
plt.figure(figsize=(12, 6))

maturity_dates = []
average_yields = []
min_yields = []
max_yields = []

for index, row in spot_rate_stored.iterrows():
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
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside

# Display plot
plt.tight_layout() 
plt.savefig("spot_rates_plot.png", dpi=300, bbox_inches="tight")  # Save plot to PNG
#plt.show()

# Forward_rates_data
forward_rate_stored = spot_rate_stored.copy()
forward_rate_stored = forward_rate_stored[forward_rate_stored.index % 2 == 0]

# Print the resulting DataFrame
# print(forward_rates_data)

def calculate_forward_rates(forward_rates_stored):
    for i, row in forward_rates_stored.iterrows():
        for col in forward_rates_stored.columns[1:-1]:
            if i != 8:
                forward_rates_stored.at[i + 2, col] = ((i / 2 + 2) * forward_rates_stored.at[i + 2, col] - forward_rates_stored.at[0, col]) / (i / 2 + 1)
    forward_rates_stored = forward_rates_stored.drop(forward_rates_stored.index[0]).reset_index(drop=True)
    return forward_rates_stored

forward_rate_stored = calculate_forward_rates(forward_rate_stored)
#print(forward_rates_data)

plt.figure(figsize=(12, 6))

# Store averages, min, and max for line plot
maturity_dates = []
average_yields = []
min_yields = []
max_yields = []

for index, row in forward_rate_stored.iterrows():
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
plt.title("Forwards Rates Over Maturity")
plt.xticks(rotation=45)
plt.grid(True)

plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside

# Display plot
plt.tight_layout()  # Adjust layout
plt.savefig("forward_rates_plot.png", dpi=300, bbox_inches="tight")  # Save plot to PNG
#plt.show()

def cov_matrix(yield_df,forward_rate_df):
    yield_stored = yield_df.drop(columns=["Date", "Maturity Date"]).values
    forward_data = forward_rate_df.drop(columns=["Date", "Maturity Date"]).values


    log_returns_yield = np.log(yield_stored[:, 1:] / yield_stored[:, :-1])
    cov_matrix_yield = np.cov(log_returns_yield)


    log_returns_forward = np.log(forward_data[:, 1:] / forward_data[:, :-1])
    cov_matrix_forward = np.cov(log_returns_forward)

    cov_df_yield = pd.DataFrame(cov_matrix_yield, 
                                index=yield_df["Maturity Date"], 
                                columns=yield_df["Maturity Date"])

    cov_df_forward = pd.DataFrame(cov_matrix_forward, 
                                index=forward_rate_df["Maturity Date"], 
                                columns=forward_rate_df["Maturity Date"])

    # Output the covariance matrices
    print("Covariance Matrix for Yield Rates:")
    cov_df_yield.to_excel("Cov_Yield.xlsx")
    print(cov_df_yield)

    print("\nCovariance Matrix for Forward Rates:")
    cov_df_forward.to_excel("Cov_Forward.xlsx")
    print(cov_df_forward)
    return cov_df_yield, cov_df_forward

cov_y, cov_f =cov_matrix(cov_yield_data,forward_rate_stored)

# Calculate eigenvalues and eigenvectors for yield covariance matrix
eigenval_yield, eigenvec_yield = np.linalg.eig(cov_y.values)
eigenval_forward, eigenvec_forward = np.linalg.eig(cov_f.values)


print("Eigenvalues of Yield Covariance Matrix:")
print(eigenval_yield)

print("\nEigenvectors of Yield Covariance Matrix:")
print(eigenvec_yield)

print("\nEigenvalues of Forward Rate Covariance Matrix:")
print(eigenval_forward)

print("\nEigenvectors of Forward Rate Covariance Matrix:")
print(eigenvec_forward)