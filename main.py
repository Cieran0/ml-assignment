# Imports for python script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Random state parameter
GLOBAL_RANDOM_STATE = 67

# Load the dataset
df = pd.read_csv("hurricane.csv")

# Display first 5 rows
df.head()

# Histogram of Status (as per assignment)
status_counts = df['Status'].value_counts()
plt.figure(figsize=(10, 6))
ax = status_counts.plot(kind='bar')
plt.title('Frequency of Cyclone Statuses')
plt.xlabel('Status')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
for i, count in enumerate(status_counts):
    ax.text(i, count, str(count), ha='center', va='bottom')
plt.show()

# Remove UNNAMED storms
df = df[df['Name'] != 'UNNAMED'].copy()

# Define numerical columns based on actual column names
numerical_cols = [
    'Maximum Wind', 'Minimum Pressure', 'Radius of Maximum Wind (RMW)',
    'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW'
]

for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.replace(-999.0, np.nan, inplace=True)

# Parse latitude and longitude
def parse_coordinate(coord):
    if pd.isna(coord) or not isinstance(coord, str):
        return np.nan
    coord = coord.strip().upper()
    if not coord:
        return np.nan
    if coord[-1] in ['N', 'S', 'E', 'W']:
        try:
            val = float(coord[:-1])
            return -val if coord[-1] in ['S', 'W'] else val
        except:
            return np.nan
    else:
        try:
            return float(coord)
        except:
            return np.nan

df['latitude_num'] = df['Latitude'].apply(parse_coordinate)
df['longitude_num'] = df['Longitude'].apply(parse_coordinate)

# Parse date and time
df['Date'] = df['Date'].astype(str)
df['Time'] = df['Time'].astype(str).str.zfill(4)

valid_mask = df['Date'].str.match(r'^\d{8}$') & df['Time'].str.match(r'^[0-2]\d[0-5]\d$')
df = df[valid_mask].copy()

df['year'] = df['Date'].str[:4].astype(int)
df['month'] = df['Date'].str[4:6].astype(int)
df['day'] = df['Date'].str[6:8].astype(int)
df['hour'] = df['Time'].str[:2].astype(int)

# Handle missingness
missing_pct = df.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 80].index.tolist()
df = df.drop(columns=cols_to_drop)

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Feature selection
feature_cols = [
    'Maximum Wind', 'Minimum Pressure', 'Radius of Maximum Wind (RMW)',
    'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW',
    'latitude_num', 'longitude_num',
    'year', 'month', 'day', 'hour'
]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df['Status'].copy()

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Plot histograms for key features
key_features = ['Maximum Wind', 'Minimum Pressure', 'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    if feat in df.columns:
        sns.histplot(df[feat].dropna(), ax=axes[i])
        axes[i].set_title(f'Distribution of {feat}')
    else:
        axes[i].text(0.5, 0.5, 'Missing', ha='center', transform=axes[i].transAxes)
plt.tight_layout()
plt.show()

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=GLOBAL_RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=GLOBAL_RANDOM_STATE)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Random Forest Training Curves
n_estimators_range = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 250, 300]
train_accuracies = []
val_accuracies = []

# Compute class weights once from full training labels
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

rf_curve = RandomForestClassifier(
    n_estimators=1,
    class_weight=class_weight_dict,
    warm_start=True,
    random_state=GLOBAL_RANDOM_STATE,
    n_jobs=-1
)

for n in n_estimators_range:
    rf_curve.n_estimators = n
    rf_curve.fit(X_train, y_train)
    train_accuracies.append(rf_curve.score(X_train, y_train))
    val_accuracies.append(rf_curve.score(X_val, y_val))

plt.figure(figsize=(12, 6))
plt.plot(n_estimators_range, train_accuracies, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(n_estimators_range, val_accuracies, 'o-', label='Validation Accuracy', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest Training Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Random Forest model with all estimators
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=GLOBAL_RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Feature importances
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='importance', y='feature')
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# RF Confusion Matrix
rf_pred = rf.predict(X_test)
cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Per class accuracy for RF
per_class_acc_rf = cm_rf.diagonal() / cm_rf.sum(axis=1)
per_class_df_rf = pd.DataFrame({
    'Class': le.classes_,
    'Accuracy': per_class_acc_rf
})

print("\nPer-class Accuracy for Random Forest:")
print(per_class_df_rf.to_string(index=False))
plt.figure(figsize=(12, 6))
sns.barplot(data=per_class_df_rf, x='Class', y='Accuracy')
plt.title('Per-class Accuracy - Random Forest')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# Example Decision Tree (Regular data)
example_tree_regular = rf.estimators_[0]

plt.figure(figsize=(40, 20))
plot_tree(example_tree_regular, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True, fontsize=4, max_depth=10)
plt.title("Example Decision Tree Regular Data", fontsize=24)
plt.show()

# Neural Network
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

nn_model.compile(optimizer='adam', 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])

history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)],
    verbose=1
)

# Neural Network training curves
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(history.history['accuracy'], 'o-', color=color, label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], 'o-', color='darkblue', label='Validation Accuracy', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, 1.05)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)  
ax2.plot(history.history['loss'], 's--', color=color, label='Train Loss', linewidth=2)
ax2.plot(history.history['val_loss'], 's--', color='darkred', label='Validation Loss', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, max(history.history['loss'] + history.history['val_loss']) * 1.1)

fig.tight_layout()
plt.title('Neural Network Training Curves')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.show()

nn_pred = np.argmax(nn_model.predict(X_test, verbose=0), axis=1)
cm_nn = confusion_matrix(y_test, nn_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap="Purples", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Neural Network Confusion Matrix")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Per-class accuracy for NN
per_class_acc_nn = cm_nn.diagonal() / cm_nn.sum(axis=1)
per_class_df_nn = pd.DataFrame({
    'Class': le.classes_,
    'Accuracy': per_class_acc_nn
})

print("\nPer-class Accuracy for Neural Network:")
print(per_class_df_nn.to_string(index=False))
plt.figure(figsize=(12, 6))
sns.barplot(data=per_class_df_nn, x='Class', y='Accuracy')
plt.title('Per-class Accuracy - Neural Network')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

train_loss, train_acc = nn_model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = nn_model.evaluate(X_val, y_val, verbose=0)

print("\nFinal Neural Network Performance:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Train Loss:     {train_loss:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
print(f"Val Loss:       {val_loss:.4f}")

# Class Imbalance Handling
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=GLOBAL_RANDOM_STATE)
under = RandomUnderSampler(sampling_strategy='auto', random_state=GLOBAL_RANDOM_STATE)
pipeline = Pipeline([("smote", smote), ("under", under)])

X_bal, y_bal = pipeline.fit_resample(X_train, y_train)

# Balanced RF
rf_bal = RandomForestClassifier(n_estimators=300, random_state=GLOBAL_RANDOM_STATE, n_jobs=-1)
rf_bal.fit(X_bal, y_bal)
rf_bal_pred = rf_bal.predict(X_test)

# Example Decision Tree (Balanced Data)
example_tree_balanced = rf_bal.estimators_[0]

plt.figure(figsize=(40, 20))
plot_tree(example_tree_balanced, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True, fontsize=4, max_depth=10)
plt.title("Example Decision Tree – Balanced Data", fontsize=24)
plt.show()

# Balanced NN
nn_bal = Sequential([
    Dense(128, activation='relu', input_shape=(X_bal.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

nn_bal.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_bal = nn_bal.fit(
    X_bal, y_bal,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# Comparison of validation accuracy between regular and balanced NN
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_accuracy'], 'o-', label='NN Regular Val Accuracy', linewidth=2)
plt.plot(history_bal.history['val_accuracy'], 'o-', label='NN Balanced Val Accuracy', linewidth=2)
plt.title("Neural Network Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Final model accuracies
rf_acc = accuracy_score(y_test, rf_pred)
nn_acc = accuracy_score(y_test, nn_pred)
rf_bal_acc = accuracy_score(y_test, rf_bal_pred)
nn_bal_pred = np.argmax(nn_bal.predict(X_test, verbose=0), axis=1)
nn_bal_acc = accuracy_score(y_test, nn_bal_pred)

# Bar chart of all model accuracies
model_names = ["RF Regular", "NN Regular", "RF Balanced", "NN Balanced"]
accuracies = [rf_acc, nn_acc, rf_bal_acc, nn_bal_acc]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=['blue', 'purple', 'skyblue', 'violet'])
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison Across Models")
plt.ylim(0, 1.05)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# Final comprehensive classification reports
print("\n=== FINAL MODEL PERFORMANCE ON TEST SET ===")
print("\nRandom Forest (Regular) Classification Report:")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

print("\nNeural Network (Regular) Classification Report:")
print(classification_report(y_test, nn_pred, target_names=le.classes_))

print("\nRandom Forest (Balanced) Classification Report:")
print(classification_report(y_test, rf_bal_pred, target_names=le.classes_))

print("\nNeural Network (Balanced) Classification Report:")
print(classification_report(y_test, nn_bal_pred, target_names=le.classes_))

# Calculate per-class accuracy for RF Balanced
cm_rf_bal = confusion_matrix(y_test, rf_bal_pred)
per_class_acc_rf_bal = cm_rf_bal.diagonal() / cm_rf_bal.sum(axis=1)

# Calculate per-class accuracy for NN Balanced
cm_nn_bal = confusion_matrix(y_test, nn_bal_pred)
per_class_acc_nn_bal = cm_nn_bal.diagonal() / cm_nn_bal.sum(axis=1)

# Create a DataFrame with all per-class accuracies
per_class_comparison = pd.DataFrame({
    'Class': le.classes_,
    'RF_Regular': per_class_acc_rf,
    'NN_Regular': per_class_acc_nn,
    'RF_Balanced': per_class_acc_rf_bal,
    'NN_Balanced': per_class_acc_nn_bal
})

# Create a grouped bar chart for per-class accuracy comparison
plt.figure(figsize=(16, 8))
x = np.arange(len(le.classes_))
width = 0.2

plt.bar(x - 1.5*width, per_class_comparison['RF_Regular'], width, label='RF Regular', color='blue', alpha=0.8)
plt.bar(x - 0.5*width, per_class_comparison['NN_Regular'], width, label='NN Regular', color='purple', alpha=0.8)
plt.bar(x + 0.5*width, per_class_comparison['RF_Balanced'], width, label='RF Balanced', color='skyblue', alpha=0.8)
plt.bar(x + 1.5*width, per_class_comparison['NN_Balanced'], width, label='NN Balanced', color='violet', alpha=0.8)

plt.xlabel('Cyclone Status Class', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-class Accuracy Comparison Across All Models', fontsize=14, fontweight='bold')
plt.xticks(x, le.classes_, rotation=45, ha='right', fontsize=10)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Models', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.show()

# Also create a summary table of per-class accuracies
print("\nPer-class Accuracy Comparison Table:")
print(per_class_comparison.round(4).to_string(index=False))