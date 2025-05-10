import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset directly defined in the code
data = pd.DataFrame({
    'Severity': [2, 3, 2, 2, 3, 2, 2, 3, 2, 2],
    'Start_Lat': [34.025917, 33.925057, 34.106038, 33.995784, 33.835801, 34.034565, 33.873535, 34.076401, 33.940819, 34.092916],
    'Start_Lng': [-118.779757, -117.431343, -117.888751, -117.81893, -117.836091, -118.13164, -117.998298, -117.857192, -118.133403, -117.870705],
    'End_Lat': [34.025917, 33.925057, 34.106038, 33.995784, 33.835801, 34.034565, 33.873535, 34.076401, 33.940819, 34.092916],
    'End_Lng': [-118.779757, -117.431343, -117.888751, -117.81893, -117.836091, -118.13164, -117.998298, -117.857192, -118.133403, -117.870705],
    'Distance(mi)': [0.01] * 10,
    'Temperature(F)': [53.6, 60.8, 60.8, 60.8, 60.8, 60.8, 60.8, 60.8, 60.8, 60.8],
    'Humidity(%)': [89.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0],
    'Pressure(in)': [29.64] + [30.06] * 9,
    'Visibility(mi)': [10.0] * 10,
    'Wind_Speed(mph)': [5.8] + [11.5] * 9,
    'Weather_Condition': ['Overcast'] + ['Clear'] * 9,
    'Amenity': [False] * 10,
    'Bump': [False] * 10,
    'Crossing': [False] * 10,
    'Give_Way': [False] * 10,
    'Junction': [True] * 10,
    'No_Exit': [False] * 10,
    'Railway': [False] * 10,
    'Roundabout': [False] * 10,
    'Station': [False] * 10,
    'Stop': [False] * 10,
    'Traffic_Calming': [False] * 10,
    'Traffic_Signal': [False] + [True] * 9,
    'Sunrise_Sunset': ['Night'] * 10
})

# Feature selection
features = ['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Traffic_Signal']
X = data[features]
y = data['Severity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Feature importance plot
features_importance = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, features_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
