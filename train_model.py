import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import config

# Load processed data
df = pd.read_csv(config.DATA_PATH)

# Target and features
target_col = 'AQI_Bucket'
drop_cols = ['Date']
feature_cols = df.columns.difference([target_col] + drop_cols)

X = df[feature_cols]
y = df[target_col]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=config.TEST_SIZE,
    stratify=y,
    random_state=config.RANDOM_STATE
)

# Train model
rf_clf = RandomForestClassifier(
    n_estimators=config.N_ESTIMATORS,
    random_state=config.RANDOM_STATE,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# Save model & scaler
joblib.dump(rf_clf, config.MODEL_PATH)
joblib.dump(scaler, config.SCALER_PATH)

# Evaluate
y_pred = rf_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
