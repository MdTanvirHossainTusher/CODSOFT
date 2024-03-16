import pickle

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def wrangle(filename):
  # Read csv file
  df = pd.read_csv(filename)

  # Convert categorical target value to numerical value
  if 'species' in df.columns:
    le = LabelEncoder()
    df['species_label'] = le.fit_transform(df['species'])

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


predict = make_prediction(
    'dataset/test.csv',
    'model/iris.pkl'
)
print(predict)