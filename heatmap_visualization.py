"""
Heatmap Visualizations for Micro-Seismic Detection System
Creates various heatmaps to analyze sensor data and risk patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
import joblib

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 70)
print("GENERATING HEATMAP VISUALIZATIONS")
print("=" * 70)

# Load and prepare data
df = pd.read_csv(
    "micro_seismic_data.csv",
    skiprows=1,
    header=0,
    names=["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm"]
)
df = df.rename(columns={"distance_cm": "distance", "time_ms": "timestamp"})

print(f"\n[OK] Loaded {len(df)} data points")

# Feature Engineering
df["mag_magnitude"] = np.sqrt(df["mag_x"]**2 + df["mag_y"]**2 + df["mag_z"]**2)
df["distance_change"] = df["distance"].diff().fillna(0)
df["vibration_freq"] = df["vibration"].rolling(window=5).mean().fillna(0)

# Prepare features
X = df[["distance", "distance_change", "mag_magnitude", "vibration_freq"]]
y = df["vibration"]

# Train models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

iso_model = IsolationForest(contamination=0.1, random_state=42)
iso_model.fit(X)

df["anomaly_flag"] = iso_model.predict(X)

# Risk index calculation
def compute_risk(row):
    risk = 0
    if row["vibration"] == 1:
        risk += 40
    if abs(row["distance_change"]) > 5:
        risk += 25
    if row["mag_magnitude"] > df["mag_magnitude"].mean():
        risk += 20
    if row["anomaly_flag"] == -1:
        risk += 30
    return min(risk, 100)

df["risk_index"] = df.apply(compute_risk, axis=1)

print("[OK] Data processed and models trained")

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Micro-Seismic Detection - Heatmap Analysis', fontsize=16, fontweight='bold', y=0.995)

# 1. Correlation Heatmap of Features
print("\n[1/5] Creating correlation heatmap...")
ax1 = plt.subplot(2, 3, 1)
correlation_data = df[["distance", "distance_change", "mag_magnitude", "vibration_freq", 
                       "mag_x", "mag_y", "mag_z", "vibration", "risk_index"]].corr()
sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1)
ax1.set_title('Feature Correlation Matrix', fontweight='bold', pad=10)

# 2. Risk Index Heatmap Over Time (2D)
print("[2/5] Creating risk index timeline heatmap...")
ax2 = plt.subplot(2, 3, 2)
# Create time bins for heatmap
time_bins = np.linspace(df["timestamp"].min(), df["timestamp"].max(), 20)
df["time_bin"] = pd.cut(df["timestamp"], bins=time_bins, labels=False)
risk_pivot = df.groupby("time_bin")["risk_index"].mean().reset_index()
risk_matrix = risk_pivot["risk_index"].values.reshape(-1, 1)
sns.heatmap(risk_matrix, yticklabels=[f"Bin {i}" for i in range(len(risk_matrix))], 
            xticklabels=["Risk"], cmap='RdYlGn_r', annot=True, fmt='.0f', 
            cbar_kws={"label": "Risk Index"}, ax=ax2)
ax2.set_title('Risk Index Over Time', fontweight='bold', pad=10)
ax2.set_ylabel('Time Bins')

# 3. Sensor Values Heatmap (Magnetometer X, Y, Z)
print("[3/5] Creating magnetometer values heatmap...")
ax3 = plt.subplot(2, 3, 3)
# Sample data for visualization (take every 10th row to avoid overcrowding)
sample_df = df.iloc[::max(1, len(df)//30)]
mag_data = sample_df[["mag_x", "mag_y", "mag_z"]].T
sns.heatmap(mag_data, cmap='viridis', cbar_kws={"label": "Magnetometer Value"}, 
            yticklabels=["Mag X", "Mag Y", "Mag Z"], ax=ax3)
ax3.set_title('Magnetometer Values Over Time', fontweight='bold', pad=10)
ax3.set_xlabel('Time Sample')

# 4. Risk Index vs Features Heatmap
print("[4/5] Creating risk vs features heatmap...")
ax4 = plt.subplot(2, 3, 4)
# Create bins for features and risk
feature_risk_data = df[["distance", "mag_magnitude", "risk_index"]].copy()
feature_risk_data["distance_bin"] = pd.cut(feature_risk_data["distance"], bins=10, labels=False)
feature_risk_data["mag_bin"] = pd.cut(feature_risk_data["mag_magnitude"], bins=10, labels=False)
risk_pivot2 = feature_risk_data.groupby(["distance_bin", "mag_bin"])["risk_index"].mean().unstack(fill_value=0)
sns.heatmap(risk_pivot2, cmap='YlOrRd', annot=False, cbar_kws={"label": "Risk Index"}, ax=ax4)
ax4.set_title('Risk Index: Distance vs Magnetometer', fontweight='bold', pad=10)
ax4.set_xlabel('Magnetometer Magnitude (binned)')
ax4.set_ylabel('Distance (binned)')

# 5. Vibration Detection Heatmap
print("[5/5] Creating vibration detection heatmap...")
ax5 = plt.subplot(2, 3, 5)
# Create bins and count vibrations
vibration_data = df[["distance", "mag_magnitude", "vibration"]].copy()
vibration_data["distance_bin"] = pd.cut(vibration_data["distance"], bins=8, labels=False)
vibration_data["mag_bin"] = pd.cut(vibration_data["mag_magnitude"], bins=8, labels=False)
vibration_pivot = vibration_data.groupby(["distance_bin", "mag_bin"])["vibration"].sum().unstack(fill_value=0)
sns.heatmap(vibration_pivot, cmap='Reds', annot=True, fmt='d', 
            cbar_kws={"label": "Vibration Count"}, ax=ax5)
ax5.set_title('Vibration Detection Frequency', fontweight='bold', pad=10)
ax5.set_xlabel('Magnetometer Magnitude (binned)')
ax5.set_ylabel('Distance (binned)')

# 6. Anomaly Detection Heatmap
ax6 = plt.subplot(2, 3, 6)
anomaly_data = df[["distance", "mag_magnitude", "anomaly_flag"]].copy()
anomaly_data["distance_bin"] = pd.cut(anomaly_data["distance"], bins=8, labels=False)
anomaly_data["mag_bin"] = pd.cut(anomaly_data["mag_magnitude"], bins=8, labels=False)
anomaly_data["anomaly_count"] = (anomaly_data["anomaly_flag"] == -1).astype(int)
anomaly_pivot = anomaly_data.groupby(["distance_bin", "mag_bin"])["anomaly_count"].sum().unstack(fill_value=0)
sns.heatmap(anomaly_pivot, cmap='Oranges', annot=True, fmt='d', 
            cbar_kws={"label": "Anomaly Count"}, ax=ax6)
ax6.set_title('Anomaly Detection Frequency', fontweight='bold', pad=10)
ax6.set_xlabel('Magnetometer Magnitude (binned)')
ax6.set_ylabel('Distance (binned)')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('heatmap_analysis.png', dpi=300, bbox_inches='tight')
print("\n[OK] Heatmap analysis saved as 'heatmap_analysis.png'")

# Create a detailed risk index heatmap
print("\n[EXTRA] Creating detailed risk index heatmap...")
fig2, ax = plt.subplots(figsize=(16, 6))

# Create a 2D heatmap of risk index over time
time_window = 20  # Number of samples per window
n_windows = len(df) // time_window
risk_matrix = []

for i in range(n_windows):
    start_idx = i * time_window
    end_idx = min((i + 1) * time_window, len(df))
    window_data = df.iloc[start_idx:end_idx]
    
    # Create feature bins for this window
    distance_bins = pd.cut(window_data["distance"], bins=5, labels=False, duplicates='drop')
    mag_bins = pd.cut(window_data["mag_magnitude"], bins=5, labels=False, duplicates='drop')
    
    # Calculate average risk for each combination
    window_risk = []
    for d_bin in range(5):
        row_risks = []
        for m_bin in range(5):
            mask = (distance_bins == d_bin) & (mag_bins == m_bin)
            if mask.any():
                row_risks.append(window_data[mask]["risk_index"].mean())
            else:
                row_risks.append(0)
        window_risk.append(row_risks)
    
    risk_matrix.append(window_risk)

risk_array = np.array(risk_matrix)
risk_array = risk_array.transpose(1, 0, 2).reshape(5, -1)  # Reshape for visualization

sns.heatmap(risk_array, cmap='RdYlGn_r', cbar_kws={"label": "Risk Index"}, 
            yticklabels=[f"Dist Bin {i+1}" for i in range(5)],
            xticklabels=[f"T{i*time_window}" for i in range(n_windows)],
            ax=ax, annot=True, fmt='.0f')
ax.set_title('Risk Index Heatmap: Distance vs Time Windows', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time Windows', fontsize=12)
ax.set_ylabel('Distance Bins', fontsize=12)

plt.tight_layout()
plt.savefig('risk_index_heatmap_detailed.png', dpi=300, bbox_inches='tight')
print("[OK] Detailed risk heatmap saved as 'risk_index_heatmap_detailed.png'")

# Create sensor correlation heatmap
print("\n[EXTRA] Creating sensor correlation heatmap...")
fig3, ax = plt.subplots(figsize=(10, 8))

sensor_corr = df[["mag_x", "mag_y", "mag_z", "distance", "vibration", 
                  "mag_magnitude", "distance_change", "vibration_freq", "risk_index"]].corr()

mask = np.triu(np.ones_like(sensor_corr, dtype=bool))  # Mask upper triangle
sns.heatmap(sensor_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Sensor Data Correlation Matrix', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('sensor_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] Sensor correlation heatmap saved as 'sensor_correlation_heatmap.png'")

print("\n" + "=" * 70)
print("[SUCCESS] All heatmaps generated successfully!")
print("=" * 70)
print("\nGenerated files:")
print("  1. heatmap_analysis.png - Comprehensive heatmap dashboard")
print("  2. risk_index_heatmap_detailed.png - Detailed risk over time")
print("  3. sensor_correlation_heatmap.png - Sensor correlation matrix")
print("\n")

# Don't show plots interactively, just save them
# plt.show()  # Commented out for non-interactive execution
print("[INFO] Plots saved. Uncomment plt.show() to display interactively.")

# ========== CONTOUR-STYLE HEATMAPS ==========
print("\n" + "=" * 70)
print("GENERATING CONTOUR-STYLE HEATMAPS")
print("=" * 70)

# Prepare data for contour plots
x_range = np.linspace(df["distance"].min(), df["distance"].max(), 50)
y_range = np.linspace(df["mag_magnitude"].min(), df["mag_magnitude"].max(), 50)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

risk_points = df[["distance", "mag_magnitude", "risk_index"]].values
Z_risk = griddata(risk_points[:, :2], risk_points[:, 2], (X_grid, Y_grid), method='cubic', fill_value=0)

# Create contour plots dashboard
fig_contour = plt.figure(figsize=(18, 12))
fig_contour.suptitle('Micro-Seismic Detection - Contour Heatmaps', fontsize=16, fontweight='bold', y=0.995)

# 1. Risk Index Contour Plot
print("\n[1/6] Creating risk index contour plot...")
ax1 = plt.subplot(2, 3, 1)
contour1 = ax1.contourf(X_grid, Y_grid, Z_risk, levels=20, cmap='RdYlGn_r', alpha=0.8)
contour_lines = ax1.contour(X_grid, Y_grid, Z_risk, levels=10, colors='black', alpha=0.3, linewidths=0.5)
ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%d')
ax1.scatter(df["distance"], df["mag_magnitude"], c=df["risk_index"], cmap='RdYlGn_r', 
            s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(contour1, ax=ax1, label='Risk Index')
ax1.set_xlabel('Distance (cm)', fontsize=10)
ax1.set_ylabel('Magnetometer Magnitude', fontsize=10)
ax1.set_title('Risk Index Contour Map', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3)

# 2. Vibration Detection Contour
print("[2/6] Creating vibration detection contour plot...")
ax2 = plt.subplot(2, 3, 2)
vibration_points = df[df["vibration"] == 1][["distance", "mag_magnitude"]].values
if len(vibration_points) > 1:
    kde = gaussian_kde(vibration_points.T)
    Z_vib = kde(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)
    contour2 = ax2.contourf(X_grid, Y_grid, Z_vib, levels=15, cmap='Reds', alpha=0.8)
    contour_lines2 = ax2.contour(X_grid, Y_grid, Z_vib, levels=8, colors='darkred', alpha=0.4, linewidths=0.5)
    ax2.clabel(contour_lines2, inline=True, fontsize=8, fmt='%.2f')
    plt.colorbar(contour2, ax=ax2, label='Vibration Density')
else:
    ax2.text(0.5, 0.5, 'No vibration data\nfor contour', ha='center', va='center', transform=ax2.transAxes)
ax2.scatter(df[df["vibration"] == 1]["distance"], df[df["vibration"] == 1]["mag_magnitude"], 
            c='red', s=50, marker='x', linewidths=2, label='Vibrations')
ax2.set_xlabel('Distance (cm)', fontsize=10)
ax2.set_ylabel('Magnetometer Magnitude', fontsize=10)
ax2.set_title('Vibration Detection Contour', fontweight='bold', pad=10)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Anomaly Detection Contour
print("[3/6] Creating anomaly detection contour plot...")
ax3 = plt.subplot(2, 3, 3)
anomaly_points = df[df["anomaly_flag"] == -1][["distance", "mag_magnitude"]].values
if len(anomaly_points) > 1:
    kde_anom = gaussian_kde(anomaly_points.T)
    Z_anom = kde_anom(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)
    contour3 = ax3.contourf(X_grid, Y_grid, Z_anom, levels=15, cmap='Oranges', alpha=0.8)
    contour_lines3 = ax3.contour(X_grid, Y_grid, Z_anom, levels=8, colors='darkorange', alpha=0.4, linewidths=0.5)
    ax3.clabel(contour_lines3, inline=True, fontsize=8, fmt='%.2f')
    plt.colorbar(contour3, ax=ax3, label='Anomaly Density')
else:
    ax3.text(0.5, 0.5, 'No anomaly data\nfor contour', ha='center', va='center', transform=ax3.transAxes)
ax3.scatter(df[df["anomaly_flag"] == -1]["distance"], df[df["anomaly_flag"] == -1]["mag_magnitude"], 
            c='orange', s=50, marker='s', edgecolors='black', linewidths=1, label='Anomalies')
ax3.set_xlabel('Distance (cm)', fontsize=10)
ax3.set_ylabel('Magnetometer Magnitude', fontsize=10)
ax3.set_title('Anomaly Detection Contour', fontweight='bold', pad=10)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Distance Change Contour
print("[4/6] Creating distance change contour plot...")
ax4 = plt.subplot(2, 3, 4)
distance_change_points = df[["distance", "mag_magnitude", "distance_change"]].values
Z_dist_change = griddata(distance_change_points[:, :2], distance_change_points[:, 2], 
                         (X_grid, Y_grid), method='cubic', fill_value=0)
contour4 = ax4.contourf(X_grid, Y_grid, Z_dist_change, levels=20, cmap='coolwarm', alpha=0.8)
contour_lines4 = ax4.contour(X_grid, Y_grid, Z_dist_change, levels=10, colors='black', alpha=0.3, linewidths=0.5)
ax4.clabel(contour_lines4, inline=True, fontsize=8, fmt='%.1f')
ax4.scatter(df["distance"], df["mag_magnitude"], c=df["distance_change"], 
            cmap='coolwarm', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(contour4, ax=ax4, label='Distance Change (cm)')
ax4.set_xlabel('Distance (cm)', fontsize=10)
ax4.set_ylabel('Magnetometer Magnitude', fontsize=10)
ax4.set_title('Distance Change Contour', fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3)

# 5. Combined Risk Zones Contour
print("[5/6] Creating combined risk zones contour...")
ax5 = plt.subplot(2, 3, 5)
risk_zones = np.zeros_like(Z_risk)
risk_zones[Z_risk < 30] = 1
risk_zones[(Z_risk >= 30) & (Z_risk < 60)] = 2
risk_zones[(Z_risk >= 60) & (Z_risk < 80)] = 3
risk_zones[Z_risk >= 80] = 4
contour5 = ax5.contourf(X_grid, Y_grid, risk_zones, levels=[0.5, 1.5, 2.5, 3.5, 4.5], 
                        colors=['green', 'yellow', 'orange', 'red'], alpha=0.7)
contour_lines5 = ax5.contour(X_grid, Y_grid, Z_risk, levels=[30, 60, 80], 
                             colors='black', linewidths=2, linestyles='--')
ax5.clabel(contour_lines5, inline=True, fontsize=10, fmt='Risk: %d')
ax5.scatter(df["distance"], df["mag_magnitude"], c=df["risk_index"], 
            cmap='RdYlGn_r', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Low Risk (0-29)'),
    Patch(facecolor='yellow', label='Medium Risk (30-59)'),
    Patch(facecolor='orange', label='High Risk (60-79)'),
    Patch(facecolor='red', label='Critical Risk (80-100)')
]
ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)
ax5.set_xlabel('Distance (cm)', fontsize=10)
ax5.set_ylabel('Magnetometer Magnitude', fontsize=10)
ax5.set_title('Risk Zones Contour Map', fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3)

# 6. 3D-style Surface Contour
print("[6/6] Creating 3D-style surface contour...")
ax6 = plt.subplot(2, 3, 6, projection='3d')
surf = ax6.plot_surface(X_grid, Y_grid, Z_risk, cmap='RdYlGn_r', alpha=0.8, 
                        linewidth=0, antialiased=True, edgecolor='none')
ax6.contour(X_grid, Y_grid, Z_risk, zdir='z', offset=Z_risk.min(), cmap='RdYlGn_r', alpha=0.5)
ax6.set_xlabel('Distance (cm)', fontsize=9)
ax6.set_ylabel('Magnetometer Magnitude', fontsize=9)
ax6.set_zlabel('Risk Index', fontsize=9)
ax6.set_title('Risk Index 3D Surface', fontweight='bold', pad=10)
fig_contour.colorbar(surf, ax=ax6, shrink=0.5, aspect=5, label='Risk Index')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('contour_heatmaps.png', dpi=300, bbox_inches='tight')
print("\n[OK] Contour heatmaps saved as 'contour_heatmaps.png'")

# Create detailed risk contour plot
print("\n[EXTRA] Creating detailed risk contour plot...")
fig_contour2, ax = plt.subplots(figsize=(14, 10))
x_hr = np.linspace(df["distance"].min(), df["distance"].max(), 100)
y_hr = np.linspace(df["mag_magnitude"].min(), df["mag_magnitude"].max(), 100)
X_hr, Y_hr = np.meshgrid(x_hr, y_hr)
Z_hr = griddata(risk_points[:, :2], risk_points[:, 2], (X_hr, Y_hr), method='cubic', fill_value=0)
contour_filled = ax.contourf(X_hr, Y_hr, Z_hr, levels=[0, 30, 60, 80, 100], 
                             colors=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'], alpha=0.7)
contour_lines_hr = ax.contour(X_hr, Y_hr, Z_hr, levels=[0, 30, 60, 80, 100], 
                              colors='black', linewidths=2, linestyles='-')
ax.clabel(contour_lines_hr, inline=True, fontsize=12, fmt='%d')
scatter = ax.scatter(df["distance"], df["mag_magnitude"], c=df["risk_index"], 
                     cmap='RdYlGn_r', s=50, alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
cbar = plt.colorbar(scatter, ax=ax, label='Risk Index', shrink=0.8)
cbar.set_label('Risk Index', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlabel('Distance (cm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Magnetometer Magnitude', fontsize=14, fontweight='bold')
ax.set_title('Detailed Risk Index Contour Map with Zones', fontsize=16, fontweight='bold', pad=20)
legend_elements_detailed = [
    plt.Rectangle((0,0),1,1, facecolor='#2ecc71', label='Low Risk (0-29)', alpha=0.7),
    plt.Rectangle((0,0),1,1, facecolor='#f1c40f', label='Medium Risk (30-59)', alpha=0.7),
    plt.Rectangle((0,0),1,1, facecolor='#e67e22', label='High Risk (60-79)', alpha=0.7),
    plt.Rectangle((0,0),1,1, facecolor='#e74c3c', label='Critical Risk (80-100)', alpha=0.7)
]
ax.legend(handles=legend_elements_detailed, loc='upper left', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig('risk_contour_detailed.png', dpi=300, bbox_inches='tight')
print("[OK] Detailed risk contour saved as 'risk_contour_detailed.png'")

print("\n" + "=" * 70)
print("[SUCCESS] All contour heatmaps generated successfully!")
print("=" * 70)
print("\nGenerated contour files:")
print("  1. contour_heatmaps.png - Comprehensive contour dashboard (6 plots)")
print("  2. risk_contour_detailed.png - Detailed risk contour with zones")
print("\n")

