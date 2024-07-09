import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv('house_prices.csv')  # Ensure your dataset is in the correct path

# Preprocessing
df.fillna(df.mean(), inplace=True)  # Handle missing values
X = df[['size', 'bedrooms', 'location']]
y = df['price']

# Dummy encoding for categorical variables
X = pd.get_dummies(X, columns=['location'], drop_first=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
