import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# -------------------------------
# Data cleaning utilities
# -------------------------------
def clean_money(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.replace("$", "").replace(",", "").replace("%", "").strip()
        if val in ["N/A", "NA", "-", ""]:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

# -------------------------------
# Tax calculation (Australian 2024–25 resident tax rates + 2% Medicare levy already baked in)
# -------------------------------
def net_tax(gross_annual):
    taxable = gross_annual * 0.98  # Approximate Medicare levy
    if gross_annual <= 18200:
        tax = 0
    elif gross_annual <= 45000:
        tax = (gross_annual - 18200) * 0.16
    elif gross_annual <= 135000:
        tax = 4288 + (gross_annual - 45000) * 0.30
    elif gross_annual <= 190000:
        tax = 31288 + (gross_annual - 120000) * 0.37
    else:
        tax = 51638 + (gross_annual - 180000) * 0.45
    return gross_annual - tax

# -------------------------------
# Main calculation: years to save for 20% deposit
# -------------------------------
def years_to_save(savings, weekly_income_net, price, income_growth, interest, property_growth, yr_max, savings_rate):
    current_price = price
    current_savings = savings
    current_income_annual_net = weekly_income_net * 52

    for year in range(1, yr_max + 1):
        annual_savings = current_income_annual_net * savings_rate
        current_savings += annual_savings
        current_savings *= (1 + interest)

        required_deposit = current_price * 0.20
        if current_savings >= required_deposit:
            return year

        # Grow everything for next year
        current_price *= (1 + property_growth)
        current_income_annual_net *= (1 + income_growth)

    return np.nan

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Property Saving Time Calculator (Interactive)")

# Load fixed CSV
file_path = "property_data.csv"
try:
    df = pd.read_csv(file_path)
    df = df.iloc[1:].reset_index(drop=True)  # Skip weird first row
    file_loaded = True
    st.success("Using built-in property dataset")
except Exception as e:
    st.error(f"Could not load data file: {e}")
    file_loaded = False

if file_loaded:
    df = df.rename(columns={
        "Unnamed: 33": "Deposit_20",
        "Unnamed: 34": "Savings_Rate",
        "Unnamed: 28": "Weekly_Net_Income",
        "MEDIAN PRICE": "Median_Price"
    })

    money_cols = ["Deposit_20", "Savings_Rate", "Weekly_Net_Income", "Median_Price"]
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_money)

    st.subheader("Model Parameters")
    savings = st.number_input("Initial Savings ($)", value=0.0, min_value=0.0, step=1000.0)
    savings_rate = st.slider("Savings Rate (fraction of net income)", 0.00, 0.50, 0.10, 0.01)
    income_growth = st.slider("Annual Income Growth Rate", 0.00, 0.10, 0.03, 0.005)
    interest = st.slider("Savings Interest Rate (p.a.)", 0.00, 0.10, 0.04, 0.005)
    property_growth = st.slider("Property Growth Rate (p.a.)", 0.00, 0.15, 0.075, 0.005)
    yr_max = st.number_input("Max Years to Simulate (don't really need to change this - this just says after the simulation as run for X years, it'll assume it's not possible and move on)", min_value=10, max_value=1000, value=250)

    st.subheader("Income Range (Pre-Tax / Gross)")
    single_income_min = st.number_input("Min Single Pre-Tax Income", value=100000.0, step=10000.0)
    single_income_max = st.number_input("Max Single Pre-Tax Income", value=500000.0, step=10000.0)
    income_step = st.number_input("Income Step (ie at default of 50k, we'll see results for 100k, 150k, 200k, ..., 500k)", value=50000.0, step=10000.0)

    # List of pre-tax incomes (this is what we display everywhere)
    pre_tax_incomes = np.arange(single_income_min, single_income_max + income_step, income_step)

    # Pre-calculate net incomes (used only internally)
    single_net_annual = [net_tax(inc) for inc in pre_tax_incomes]
    dual_net_annual   = [net_tax(inc) * 2 for inc in pre_tax_incomes]

    # For display: create nice labels
    def income_label(pre_tax, household_type):
        k = int(pre_tax / 1000)
        if household_type == "single":
            return f"${k}k Single"
        else:
            return f"${k}k Dual\n({k*2}k combined pre-tax)"

    if st.button("Run Simulation"):
        df_valid = df.dropna(subset=["Median_Price"]).copy()

        results_summary = {}
        yr_results = {}

        # Helper to run simulation for a list of net incomes
        def run_simulation(net_incomes_list, household_type):
            for pre_tax, net_annual in zip(pre_tax_incomes, net_incomes_list):
                weekly_net = net_annual / 52
                col_name = f"Years_{household_type}_{int(pre_tax/1000)}k"
                label = income_label(pre_tax, household_type)

                years_list = []
                for _, row in df_valid.iterrows():
                    years = years_to_save(
                        savings=savings,
                        weekly_income_net=weekly_net,
                        price=row["Median_Price"],
                        income_growth=income_growth,
                        interest=interest,
                        property_growth=property_growth,
                        yr_max=yr_max,
                        savings_rate=savings_rate
                    )
                    years_list.append(years)

                df_valid[col_name] = years_list

                attainable = df_valid[col_name].notna()
                results_summary[label] = {
                    "Pre-Tax Income": pre_tax,
                    "Household": household_type.capitalize(),
                    "Attainable Suburbs": attainable.sum(),
                    "Unattainable Suburbs": (~attainable).sum(),
                    "Mean Years (attainable only)": df_valid[col_name][attainable].mean(),
                    "Median Years": df_valid[col_name][attainable].median(),
                }

                yr_results[label] = df_valid[col_name][attainable]

        # Run both single and dual
        run_simulation(single_net_annual, "single")
        run_simulation(dual_net_annual,   "dual")

        # ========================
        # Histograms
        # ========================
        st.subheader("Distribution of Years to Save")
        n_plots = len(yr_results)
        cols = 3
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, (label, data) in enumerate(yr_results.items()):
            ax = axes[idx]
            ax.hist(data, bins=30, edgecolor='black', color='skyblue', alpha=0.8)
            unattainable = results_summary[label]["Unattainable Suburbs"]
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.set_xlabel("Years to Save")
            ax.set_ylabel("Number of Suburbs")
            ax.text(0.95, 0.95, f"Unattainable: {unattainable}",
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle="round", facecolor="wheat"))

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        # ========================
        # Summary Table
        # ========================
        st.subheader("Summary Table (All figures based on Pre-Tax Income)")
        summary_df = pd.DataFrame(results_summary).T
        summary_df = summary_df[["Pre-Tax Income", "Household", "Attainable Suburbs",
                                 "Unattainable Suburbs", "Mean Years (attainable only)", "Median Years"]]
        st.dataframe(summary_df.sort_values("Pre-Tax Income"))

        # ========================
        # Unattainable Suburbs Bar Chart
        # ========================
        st.subheader("Unattainable Suburbs by Pre-Tax Income")
        total_suburbs = len(df_valid)

        labels = list(results_summary.keys())
        unattainable_counts = [results_summary[l]["Unattainable Suburbs"] for l in labels]

        fig2, ax2 = plt.subplots(figsize=(14, 7))
        bars = ax2.bar(labels, unattainable_counts,
                       color=['#66b3ff' if 'Single' in l else '#ff9999' for l in labels],
                       edgecolor='black')

        ax2.set_ylabel("Number of Unattainable Suburbs")
        ax2.set_title("How Many Suburbs Are Out of Reach?", fontsize=16, fontweight='bold')
        ax2.set_ylim(0, total_suburbs + 50)
        ax2.axhline(y=total_suburbs, color='red', linestyle='--', linewidth=2,
                    label=f"All {total_suburbs} suburbs")
        ax2.legend()

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)

        # ========================
        # Extended Stats Table
        # ========================
        st.subheader("Detailed Statistics (Attainable Suburbs Only)")
        extra = {}
        for label, data in yr_results.items():
            extra[label] = {
                "Mean Years": data.mean(),
                "Median Years": data.median(),
                "Std Dev": data.std(),
                "Min Years": data.min(),
                "Max Years": data.max(),
                "10th Percentile": data.quantile(0.1),
                "90th Percentile": data.quantile(0.9),
            }
        st.dataframe(pd.DataFrame(extra).T.round(2))
