import pandas as pd
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load data
df = pd.read_csv('recipes.csv')

# Drop NA
df.dropna(inplace=True)

# Encode user_id and recipe_id
user_enc = LabelEncoder()
recipe_enc = LabelEncoder()

df['user_enc'] = user_enc.fit_transform(df['user_id'])
df['recipe_enc'] = recipe_enc.fit_transform(df['recipe_id'])

# Surprise requires only user, item, rating
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['user_enc', 'recipe_enc', 'rating']], reader)

# Train/Test Split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'SVD': SVD(),
    'KNNBasic': KNNBasic()
}

best_rmse = float('inf')
best_model = None
best_name = ''

for name, model in models.items():
    model.fit(trainset)
    predictions = model.test(testset)
    y_true = [pred.r_ui for pred in predictions]
    y_pred = [pred.est for pred in predictions]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} RMSE: {rmse:.3f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'user_encoder': user_enc,
        'recipe_encoder': recipe_enc,
        'df': df
    }, f)

print(f"Saved best model: {best_name} (RMSE: {best_rmse:.3f})")
