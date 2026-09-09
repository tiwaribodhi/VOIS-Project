# ============================================================
# Seasonal Agriculture Performance Analysis
# VOIS AICTE Batch 1 | Major Project 2026-2027
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = "seasonal_agriculture_performance_dataset.csv"

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("SEASONAL AGRICULTURE PERFORMANCE ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nMissing Values by Column:")
print(missing_values[missing_values > 0])

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Numerical columns containing missing values
numeric_missing_columns = [
    "Rainfall_mm",
    "Soil_Moisture_pct",
    "Yield_Tonnes_Ha"
]

# Replace missing values with the respective median
for column in numeric_missing_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())


# ============================================================
# 5. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())


# ============================================================
# 6. CATEGORICAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL ANALYSIS")
print("=" * 60)

print("\nNumber of States:")
print(df["State"].nunique())

print("\nStates:")
print(df["State"].unique())

print("\nNumber of Crops:")
print(df["Crop"].nunique())

print("\nCrops:")
print(df["Crop"].unique())

print("\nSeasons:")
print(df["Season"].value_counts())

print("\nIrrigation Methods:")
print(df["Irrigation_Method"].value_counts())


# ============================================================
# 7. SEASONAL PERFORMANCE ANALYSIS
# ============================================================

season_order = ["Kharif", "Rabi", "Zaid"]

df["Season"] = pd.Categorical(
    df["Season"],
    categories=season_order,
    ordered=True
)

season_summary = df.groupby(
    "Season",
    observed=True
).agg(
    Farms=("Farm_ID", "count"),
    Median_Yield=("Yield_Tonnes_Ha", "median"),
    Average_Yield=("Yield_Tonnes_Ha", "mean"),
    Median_Profit=("Profit_INR", "median"),
    Average_Profit=("Profit_INR", "mean"),
    Median_Revenue=("Revenue_INR", "median"),
    Median_Rainfall=("Rainfall_mm", "median"),
    Median_Temperature=("Avg_Temperature_C", "median"),
    Median_Soil_Moisture=("Soil_Moisture_pct", "median"),
    Median_Water_Efficiency=(
        "Water_Efficiency_t_per_1000m3",
        "median"
    )
)

print("\n" + "=" * 60)
print("SEASONAL PERFORMANCE")
print("=" * 60)

print(season_summary)


# ============================================================
# 8. PROFITABILITY BY SEASON
# ============================================================

positive_profit_rate = df.groupby(
    "Season",
    observed=True
)["Profit_INR"].apply(
    lambda x: (x > 0).mean() * 100
)

print("\nPercentage of Farms with Positive Profit:")
print(positive_profit_rate)


# ============================================================
# 9. CROP-WISE PERFORMANCE
# ============================================================

crop_summary = df.groupby("Crop").agg(
    Farms=("Farm_ID", "count"),
    Median_Yield=("Yield_Tonnes_Ha", "median"),
    Average_Yield=("Yield_Tonnes_Ha", "mean"),
    Median_Profit=("Profit_INR", "median"),
    Average_Profit=("Profit_INR", "mean"),
    Median_Revenue=("Revenue_INR", "median")
).sort_values(
    "Median_Yield",
    ascending=False
)

print("\n" + "=" * 60)
print("CROP-WISE PERFORMANCE")
print("=" * 60)

print(crop_summary)


# ============================================================
# 10. CROP × SEASON ANALYSIS
# ============================================================

crop_season = df.pivot_table(
    index="Crop",
    columns="Season",
    values="Yield_Tonnes_Ha",
    aggfunc="median",
    observed=False
)

print("\n" + "=" * 60)
print("CROP × SEASON MEDIAN YIELD")
print("=" * 60)

print(crop_season)


# ============================================================
# 11. IRRIGATION ANALYSIS
# ============================================================

irrigation_summary = df.groupby(
    "Irrigation_Method"
).agg(
    Farms=("Farm_ID", "count"),
    Median_Yield=("Yield_Tonnes_Ha", "median"),
    Average_Yield=("Yield_Tonnes_Ha", "mean"),
    Median_Profit=("Profit_INR", "median"),
    Average_Profit=("Profit_INR", "mean"),
    Median_Water_Efficiency=(
        "Water_Efficiency_t_per_1000m3",
        "median"
    )
).sort_values(
    "Median_Yield",
    ascending=False
)

print("\n" + "=" * 60)
print("IRRIGATION PERFORMANCE")
print("=" * 60)

print(irrigation_summary)


# ============================================================
# 12. STATE-WISE PROFITABILITY
# ============================================================

state_summary = df.groupby("State").agg(
    Farms=("Farm_ID", "count"),
    Median_Profit=("Profit_INR", "median"),
    Average_Profit=("Profit_INR", "mean"),
    Median_Yield=("Yield_Tonnes_Ha", "median"),
    Positive_Profit_Rate=(
        "Profit_INR",
        lambda x: (x > 0).mean() * 100
    )
).sort_values(
    "Median_Profit",
    ascending=False
)

print("\n" + "=" * 60)
print("STATE-WISE PERFORMANCE")
print("=" * 60)

print(state_summary)


# ============================================================
# 13. WATER EFFICIENCY ANALYSIS
# ============================================================

water_summary = df.groupby(
    "Irrigation_Method"
).agg(
    Median_Yield=("Yield_Tonnes_Ha", "median"),
    Median_Water_Efficiency=(
        "Water_Efficiency_t_per_1000m3",
        "median"
    ),
    Median_Water_Used=("Water_Used_m3", "median")
).sort_values(
    "Median_Water_Efficiency",
    ascending=False
)

print("\n" + "=" * 60)
print("WATER EFFICIENCY ANALYSIS")
print("=" * 60)

print(water_summary)


# ============================================================
# 14. CORRELATION ANALYSIS
# ============================================================

correlation = df[
    [
        "Yield_Tonnes_Ha",
        "Rainfall_mm",
        "Avg_Temperature_C",
        "Soil_Moisture_pct",
        "Fertilizer_kg_ha",
        "Pesticide_Litre_ha",
        "Seed_Quality_Score",
        "Water_Used_m3",
        "Water_Efficiency_t_per_1000m3",
        "Disease_Pest_Risk_pct",
        "Profit_INR"
    ]
].corr()

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

print(correlation.round(2))


yield_water_correlation = df[
    "Yield_Tonnes_Ha"
].corr(
    df["Water_Efficiency_t_per_1000m3"]
)

print(
    "\nCorrelation between Yield and Water Efficiency:",
    round(yield_water_correlation, 3)
)


# ============================================================
# 15. VISUALIZATION 1
#    MEDIAN YIELD BY SEASON
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    season_summary.index.astype(str),
    season_summary["Median_Yield"]
)

plt.title("Median Agricultural Yield by Season")
plt.xlabel("Season")
plt.ylabel("Median Yield (Tonnes/Ha)")
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(season_summary["Median_Yield"]):
    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 16. VISUALIZATION 2
#    PROFITABILITY BY SEASON
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    positive_profit_rate.index.astype(str),
    positive_profit_rate.values
)

plt.title("Positive Profit Rate by Season")
plt.xlabel("Season")
plt.ylabel("Farms with Positive Profit (%)")
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(positive_profit_rate.values):
    plt.text(
        i,
        value,
        f"{value:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 17. VISUALIZATION 3
#    CROP × SEASON HEATMAP
# ============================================================

plt.figure(figsize=(9, 6))

plt.imshow(
    crop_season.values,
    aspect="auto"
)

plt.xticks(
    range(len(crop_season.columns)),
    crop_season.columns.astype(str)
)

plt.yticks(
    range(len(crop_season.index)),
    crop_season.index
)

plt.title("Median Crop Yield Across Seasons")
plt.xlabel("Season")
plt.ylabel("Crop")

for i in range(crop_season.shape[0]):
    for j in range(crop_season.shape[1]):
        value = crop_season.iloc[i, j]

        if not pd.isna(value):
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

plt.colorbar(
    label="Median Yield (Tonnes/Ha)"
)

plt.tight_layout()
plt.show()


# ============================================================
# 18. VISUALIZATION 4
#    IRRIGATION VS YIELD
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    irrigation_summary.index,
    irrigation_summary["Median_Yield"]
)

plt.title("Median Yield by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Median Yield (Tonnes/Ha)")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(
    irrigation_summary["Median_Yield"]
):
    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 19. VISUALIZATION 5
#    STATE-WISE PROFITABILITY
# ============================================================

state_plot = state_summary.sort_values(
    "Median_Profit"
)

plt.figure(figsize=(10, 6))

plt.barh(
    state_plot.index,
    state_plot["Median_Profit"]
)

plt.title("Median Profit by State")
plt.xlabel("Median Profit (INR)")
plt.ylabel("State")
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 20. VISUALIZATION 6
#    YIELD VS WATER EFFICIENCY
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Water_Efficiency_t_per_1000m3"],
    df["Yield_Tonnes_Ha"],
    alpha=0.35
)

# Trend line
x = df["Water_Efficiency_t_per_1000m3"]
y = df["Yield_Tonnes_Ha"]

slope, intercept = np.polyfit(x, y, 1)

x_line = np.linspace(
    x.min(),
    x.max(),
    100
)

y_line = slope * x_line + intercept

plt.plot(
    x_line,
    y_line
)

plt.title(
    f"Yield vs Water Efficiency "
    f"(Correlation = {yield_water_correlation:.2f})"
)

plt.xlabel(
    "Water Efficiency (Tonnes per 1000 m³)"
)

plt.ylabel(
    "Yield (Tonnes/Ha)"
)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 21. KEY FINDINGS
# ============================================================

highest_yield_season = season_summary[
    "Median_Yield"
].idxmax()

lowest_yield_season = season_summary[
    "Median_Yield"
].idxmin()

highest_profit_season = positive_profit_rate.idxmax()

best_irrigation = irrigation_summary[
    "Median_Yield"
].idxmax()

best_state = state_summary[
    "Median_Profit"
].idxmax()

best_crop = crop_summary[
    "Median_Yield"
].idxmax()


print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print(
    f"\n1. Highest median-yield season: "
    f"{highest_yield_season}"
)

print(
    f"   Median yield: "
    f"{season_summary.loc[highest_yield_season, 'Median_Yield']:.2f} tonnes/ha"
)

print(
    f"\n2. Lowest median-yield season: "
    f"{lowest_yield_season}"
)

print(
    f"   Median yield: "
    f"{season_summary.loc[lowest_yield_season, 'Median_Yield']:.2f} tonnes/ha"
)

print(
    f"\n3. Season with highest positive-profit rate: "
    f"{highest_profit_season}"
)

print(
    f"   Positive-profit farms: "
    f"{positive_profit_rate.loc[highest_profit_season]:.2f}%"
)

print(
    f"\n4. Highest-yield irrigation method: "
    f"{best_irrigation}"
)

print(
    f"   Median yield: "
    f"{irrigation_summary.loc[best_irrigation, 'Median_Yield']:.2f} tonnes/ha"
)

print(
    f"\n5. Most profitable state by median profit: "
    f"{best_state}"
)

print(
    f"   Median profit: "
    f"₹{state_summary.loc[best_state, 'Median_Profit']:,.2f}"
)

print(
    f"\n6. Highest-yield crop: "
    f"{best_crop}"
)

print(
    f"   Median yield: "
    f"{crop_summary.loc[best_crop, 'Median_Yield']:.2f} tonnes/ha"
)

print(
    f"\n7. Yield vs water-efficiency correlation: "
    f"{yield_water_correlation:.3f}"
)


# ============================================================
# 22. CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
The analysis demonstrates that agricultural performance varies
across seasons, crops, irrigation methods and geographical areas.

Seasonal comparisons reveal differences in yield, profitability,
rainfall, temperature, soil moisture and water efficiency.

Crop-level analysis helps identify crops with consistently higher
production performance, while irrigation analysis highlights
differences in yield and economic outcomes between irrigation
methods.

State-level analysis further demonstrates that agricultural
profitability is not uniform across geographical regions.

Overall, the analysis provides evidence-based insights that can
support seasonal agricultural planning, resource management and
future data-driven agricultural decision making.

The observed relationships are descriptive and should not be
interpreted as proof of causation.
""")

print("\nAnalysis completed successfully.")
