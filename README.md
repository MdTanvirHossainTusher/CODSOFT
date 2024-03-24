# Credit-Card-Fraud-Detection

A binary classification model that can classify whether a credit card data is fraudulent or legit.


# Data Collection

Data has already available [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset contains transactions made by credit cards in September 2013 by European cardholders. Dataset presents transactions that occurred in two days, which contains 492 frauds transactions out of 284,807 transactions. The dataset is highly imbalanced.


# Data Preprocessing

Initially datasets contains 1081 duplicate rows. After removing those it reduces to 283,726 observations, where positive (fraudulent) class contains only 0.001667% of data and negative class contains 0.998333% of data. To reduce imbalanced property, under and over sampling has done where over sampling performs well. On the other hand, under sampling perfomrs poorly.


# Model Training

Datasets is trained using `Decision Tree`, `Random Forest` and `Gradient Boosting` classification models. Showing each models performance, their training time and lackings.


# Result Analysis
In the table I showed the Precision, Recall, F1 score and  accuracy for three models.
<table>
<thead>
    <tr>
      <th>Model</th>
      <th colspan="2">Precision</th>
      <th colspan="2">Recall</th>
      <th colspan="2">F1-score</th>
      <th colspan="2">Accuracy</th>
    </tr>
    <tr>
      <th></th>
      <th>fraudulent</th>
      <th>Non-fraudulent</th>
      <th>fraudulent</th>
      <th>Non-fraudulent</th>
      <th>fraudulent</th>
      <th>Non-fraudulent</th>
      <th>Training</th>
      <th>Testing</th>
    </tr>
  </thead>
<tbody>
  <tr>
    <td>Decision Tree</td>
    <td>0.53</td>
    <td>1.00</td>
    <td>0.53</td>
    <td>1.00</td>
    <td>0.53</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>0.9993832055687726</td>
  </tr>
  
  <tr>
    <td>Random Forest</td>
    <td>0.94</td>
    <td>1.00</td>
    <td>0.53</td>
    <td>1.00</td>
    <td>0.68</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>0.9997</td>
  </tr>
  
  <tr>
    <td>Gradient Boosting</td>
    <td>0.03</td>
    <td>1.00</td>
    <td>0.80</td>
    <td>0.99</td>
    <td>0.07</td>
    <td>0.99</td>
    <td>0.9849</td>
    <td>0.985</td>
  </tr>
  </tbody>
</table>

Most important feature for training `Decision Tree`, `Random Forest` and `Gredient Boosting` is `V14`.

`Random Forest` is fitting 2 folds for each of 12 candidates, totalling 24 fits. On the other hand, `Gredient Boosting` is fitting 2 folds for each of 9 candidates, totalling 18 fits.

`Random Forest` performed best for `max depth = 30` and `n estimators = 75` where `n estimators` was `25, 50, 75` and `max depth` was `10, 20, 30, 40`. On the other hand, `Gredient Boosting` performed best for `max depth = 4` and `n estimators = 30` where `n estimators` was `20, 25, 30` and `max depth` was `2, 3, 4`.


`Mean Test Score` of `Gradient Boosting` is lower than `Random Forest`. But 
`Mean Fit Time` of `Gradient Boosting` is higher than `Random Forest`.

