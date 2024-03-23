import pickle
import pandas as pd



def wrangle(filepath):
  # Read csv file
  df = pd.read_csv(filepath)

  # keep the first occurrence of each duplicate row
  df.drop_duplicates(inplace=True, keep='first')

  # Get the features from `V1` to `V28`
  features = df.columns[1:29]

  # Remove outliers and get the value betweeen 0.1 to 0.9 quantile
  for feature in features: 
    q1, q9 = df[feature].quantile([0.1, 0.9])
    quantile_mask = df[feature].between(q1, q9)

  df = df[quantile_mask]

  # Reset index after dropping rows
  df.reset_index(drop=True, inplace=True)

  return df



def make_prediction(data_filepath, model_filepath):
  X_test = wrangle(data_filepath)

  # load model
  with open(model_filepath, 'rb') as f:
    model = pickle.load(f)

  y_test_pred = model.predict(X_test)
  y_test_pred = pd.Series(y_test_pred)
  return y_test_pred


dt_pred = make_prediction(
    'dataset/creditcard_test_data.csv',
    'model/dt_over.pkl'
)
print(dt_pred)
