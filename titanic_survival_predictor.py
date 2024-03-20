import pickle
import pandas as pd


def wrangle(filepath):
  # Read csv file
  df = pd.read_csv(filepath)

  # Drop null column
  df.drop(columns=['Cabin'], inplace=True)

  # drop low-high cardinality features
  df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Embarked'], inplace=True)

  # Convert `Sex` categorical to binary
  df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})

  return df


def make_prediction(data_filepath, model_filepath):
  X_test = wrangle(data_filepath)

  # load model
  with open(model_filepath, 'rb') as f:
    model = pickle.load(f)

  y_test_pred = model.predict(X_test)
  y_test_pred = pd.Series(y_test_pred)
  return y_test_pred


predict = make_prediction(
  'dataset/test.csv',
  'model/titatic_survival_lr.pkl'
)
print(predict)