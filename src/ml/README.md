# Machine Learning Models

## Isolation Forest Model

This directory should contain the pre-trained Isolation Forest model for anomaly detection.

### Training the Model

To train a new model on historical transaction data:

```python
from sklearn.ensemble import IsolationForest
import pandas as pd
import pickle

# Load historical transaction data
df = pd.read_csv('historical_transactions.csv')

# Feature engineering
df['log_amount'] = np.log1p(df['amount'])
df['vendor_encoded'] = df['vendor_id'].apply(lambda x: hash(x) % 1000)
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
df['day_of_month'] = pd.to_datetime(df['date']).dt.day

# Train model
features = ['log_amount', 'vendor_encoded', 'day_of_week', 'day_of_month']
X = df[features].fillna(0)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Save model
with open('src/ml/isolation_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Model Performance

- **Contamination rate**: 5% (expected anomaly rate)
- **Features**: log_amount, vendor_encoded, day_of_week, day_of_month
- **Training data**: Minimum 1000 transactions recommended
- **Retraining frequency**: Weekly (via feedback_analyzer.py)

### Model File

Place the trained model at: `src/ml/isolation_forest_model.pkl`

If no model exists, the system will create a default untrained model and train it on-the-fly with the first batch of transactions.
