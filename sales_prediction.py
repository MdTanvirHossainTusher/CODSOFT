import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


def wrangle(filepath):
  # Read csv file
  df = pd.read_csv(filepath)

  # Remove outliers from `Newspaper`
  low, high = df['Newspaper'].quantile([0, .98])
  mask_votes = df['Newspaper'].between(low, high)
  df = df[mask_votes]

  # Drop old index and reset new index
  df.reset_index(drop=True, inplace=True)

  return df


def make_prediction(data_filepath, model_filepath):
  X_test = wrangle(data_filepath)

  # Scaling the testing data
  scale = StandardScaler()
  X_test = scale.fit_transform(X_test)

  # load model
  with open(model_filepath, 'rb') as f:
    model = pickle.load(f)

  y_test_pred = model.predict(X_test)
  y_test_pred = pd.Series(y_test_pred)
  return y_test_pred


pred = make_prediction(
    'dataset/my_test_data.csv',
    'model/Sales_Prediction.pkl'
)
print(pred)