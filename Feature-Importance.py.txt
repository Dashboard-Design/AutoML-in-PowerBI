# 'dataset' holds the input data for this script
import pickle
import pandas as pd
# Load model from disk
file = open("E:/Python NoteBooks/model.pkl", 'rb')
model = pickle.load(file)
df = dataset
df = df.drop('Car_Name', axis=1)
df = df.drop_duplicates()
d_df = pd.get_dummies(df)
X = d_df.drop('Selling_Price', axis = 1)
df['predictions'] = model.predict(X)
feature_importances = pd.DataFrame(zip(d_df.columns, model.coef_ ), columns=['feature', 'score'])
