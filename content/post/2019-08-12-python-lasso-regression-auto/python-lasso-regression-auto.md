---
title: Lasso Regression with Python
date: '2019-08-12'
slug: python-lasso-regression
categories:
  - Statistics
tags:
  - Python
  - Regression
  - Statistics
subtitle: ''
summary: 'Implementation of Lasso Regression in Python'
authors: [Jan]
lastmod: '2019-08-12T08:30:37+02:00'
featured: no
image:
  caption: 'Lasso Regression'
  focal_point: ''
  preview_only: yes
projects: []
---


Lasso performs a so called `L1 regularization` (a process of introducing additional information in order to prevent overfitting), i.e. adds penalty equivalent to absolute value of the magnitude of coefficients.

In particular, the minimization objective does not only include the residual sum of squares (RSS) but also the sum of the absolute value of coefficients.

The residual sum of squares (RSS) is calculated as follows:

$$ RSS = \sum_{i=1}^{n} (y - \hat{y})^2 $$

Hence, the minimization objective becomes: RSS + α (sum of absolute value of coefficients), where α (alpha) provides a trade-off between balancing RSS and magnitude of coefficients.

α can take various values:

  - α = 0: Same coefficients as simple linear regression
  - α = ∞: All coefficients zero (same logic as before)
  - 0 < α < ∞: coefficients between 0 and that of simple linear regression

---

## Data preparation

This tutorial involves the use of the Lasso regression on the "Auto" dataset. In particular, we only use observations 1 to 200 for our analysis. Furthermore, you can drop the `name` variable.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import warnings; warnings.simplefilter('ignore')
```


```python
df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/Auto.csv")
```

**Tidying data**


```python
df = df.iloc[0:200]
df = df.drop(['name'], axis=1)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 8 columns):
    mpg             200 non-null float64
    cylinders       200 non-null int64
    displacement    200 non-null float64
    horsepower      200 non-null object
    weight          200 non-null int64
    acceleration    200 non-null float64
    year            200 non-null int64
    origin          200 non-null int64
    dtypes: float64(3), int64(4), object(1)
    memory usage: 12.6+ KB



```python
df['origin'] = pd.Categorical(df['origin'])
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
print(df.isnull().sum())
```

    mpg             0
    cylinders       0
    displacement    0
    horsepower      2
    weight          0
    acceleration    0
    year            0
    origin          0
    dtype: int64



```python
# drop missing cases
df = df.dropna()
```


# Implementation of Lasso regression

We use scikit learn to fit a Lasso regression [(see documentation)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) and follow a number of steps:

  - (1.1) **Standardize** the features (module: `from sklearn.preprocessing import StandardScaler`)


*Hint: It is important to standardize the features by removing the mean and scaling to unit variance. The L1 (Lasso) and L2 (Ridge) regularizers of linear models assume that all features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.*


  - (1.2) **Split the data set** into train and test sets (use `X_train`, `X_test`, `y_train`, `y_test`), with the first 75% of the data for training and the remaining for testing. (module: `from sklearn.model_selection import train_test_split`)


  - (1.3) Apply **Lasso regression** on the training set with the regularization parameter **lambda = 0.5** (module: `from sklearn.linear_model import Lasso`) and print the $R^2$-score for the training and test set. Comment on your findings.


  - (1.4) Apply the **Lasso regression** on the training set with the following **λ parameters: (0.001, 0.01, 0.1, 0.5, 1, 2, 10)**. Evaluate the R^2 score for all the models you obtain on both the train and test sets.


  - (1.5) **Plot** all values for both data sets (train and test $R^2$-values) as a function of λ. Comment on your findings.


  - (1.6) Store your test data results in a DataFrame and indentify the lambda where the $R^2$ has it's **maximum value** in the **test data**. Fit a Lasso model with this lambda parameter (use the training data) and obtain the corresponding **regression coefficients**. Furthermore, obtain the **mean squared error** for the test data of this model (module: `from sklearn.metrics import mean_squared_error`)


  - (1.7) Evaluate the performance of a **Lasso regression** for different regularization parameters λ using **5-fold cross validation** on the training set (module: `from sklearn.model_selection import cross_val_score`) and plot the cross-validation (CV) $R^2$ scores of the training and test data as a function of λ.


    Use the following lambda parameters:
    l_min = 0.05
    l_max = 0.2
    l_num = 20
    lambdas = np.linspace(l_min,l_max, l_num)


  - (1.8) Finally, store your test data results in a DataFrame and identify the lambda where the $R^2$ has it's **maximum value** in the **test data**. Fit a Lasso model with this lambda parameter (use the training data) and obtain the corresponding **regression coefficients**. Furthermore, obtain the **mean squared error** for the test data of this model (module: `from sklearn.metrics import mean_squared_error`)

---


### 1.1 Standardization


```python
dfs = df.astype('int')
```


```python
dfs.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 198 entries, 0 to 199
    Data columns (total 8 columns):
    mpg             198 non-null int64
    cylinders       198 non-null int64
    displacement    198 non-null int64
    horsepower      198 non-null int64
    weight          198 non-null int64
    acceleration    198 non-null int64
    year            198 non-null int64
    origin          198 non-null int64
    dtypes: int64(8)
    memory usage: 13.9 KB



```python
dfs.columns
```




    Index(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'year', 'origin'],
          dtype='object')




```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dfs[['cylinders', 'displacement', 'horsepower',
     'weight', 'acceleration', 'year', 'origin']] = scaler.fit_transform(dfs[['cylinders',
                                                                              'displacement',
                                                                              'horsepower',
                                                                              'weight',
                                                                              'acceleration',
                                                                              'year', 'origin']])
```


```python
dfs.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>1.179744</td>
      <td>0.726091</td>
      <td>0.325216</td>
      <td>0.346138</td>
      <td>-0.955578</td>
      <td>-1.516818</td>
      <td>-0.629372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>1.179744</td>
      <td>1.100254</td>
      <td>1.129264</td>
      <td>0.548389</td>
      <td>-1.305309</td>
      <td>-1.516818</td>
      <td>-0.629372</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>1.179744</td>
      <td>0.821807</td>
      <td>0.784672</td>
      <td>0.273370</td>
      <td>-1.305309</td>
      <td>-1.516818</td>
      <td>-0.629372</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>1.179744</td>
      <td>0.699986</td>
      <td>0.784672</td>
      <td>0.270160</td>
      <td>-0.955578</td>
      <td>-1.516818</td>
      <td>-0.629372</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>1.179744</td>
      <td>0.682583</td>
      <td>0.554944</td>
      <td>0.287282</td>
      <td>-1.655041</td>
      <td>-1.516818</td>
      <td>-0.629372</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 Split data


```python
X = dfs.drop(['mpg'], axis=1)
y = dfs['mpg']
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
```

### 1.3 Lasso


```python
from sklearn.linear_model import Lasso

reg = Lasso(alpha=0.5)
reg.fit(X_train, y_train)
```




    Lasso(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
print('Lasso Regression: R^2 score on training set', reg.score(X_train, y_train)*100)
print('Lasso Regression: R^2 score on test set', reg.score(X_test, y_test)*100)
```

    Lasso Regression: R^2 score on training set 82.49741060950073
    Lasso Regression: R^2 score on test set 85.49734440925533


### 1.4 Lasso with different lambdas


```python
lambdas = (0.001, 0.01, 0.1, 0.5, 1, 2, 10)
l_num = 7
pred_num = X.shape[1]

# prepare data for enumerate
coeff_a = np.zeros((l_num, pred_num))
train_r_squared = np.zeros(l_num)
test_r_squared = np.zeros(l_num)
```


```python
# enumerate through lambdas with index and i
for ind, i in enumerate(lambdas):    
    reg = Lasso(alpha = i)
    reg.fit(X_train, y_train)

    coeff_a[ind,:] = reg.coef_
    train_r_squared[ind] = reg.score(X_train, y_train)
    test_r_squared[ind] = reg.score(X_test, y_test)
```

### 1.5 Plot


```python
# Plotting
plt.figure(figsize=(18, 8))
plt.plot(train_r_squared, 'bo-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
plt.plot(test_r_squared, 'bo-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)
plt.xlabel('Lamda index'); plt.ylabel(r'$R^2$')
plt.xlim(0, 6)
plt.title(r'Evaluate lasso regression with lamdas: 0 = 0.001, 1= 0.01, 2 = 0.1, 3 = 0.5, 4= 1, 5= 2, 6 = 10')
plt.legend(loc='best')
plt.grid()
```

![](/post/2019-08-12-python-lasso-regression-auto/output_27_0.png)



### 1.6 Identify best lambda and coefficients


```python
df_lam = pd.DataFrame(test_r_squared*100, columns=['R_squared'])
df_lam['lambda'] = (lambdas)
# returns the index of the row where column has maximum value.
df_lam.loc[df_lam['R_squared'].idxmax()]
```




    R_squared    88.105773
    lambda        0.001000
    Name: 0, dtype: float64




```python
# Coefficients of best model
reg_best = Lasso(alpha = 0.1)
reg_best.fit(X_train, y_train)
reg_best.coef_
```




    array([-0.35554113, -1.13104696, -0.00596296, -3.31741775, -0.        ,
            0.37914648,  0.74902885])




```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, reg_best.predict(X_test))
```




    3.586249592807347



### 1.7 Cross Validation


```python
l_min = 0.05
l_max = 0.2
l_num = 20
lambdas = np.linspace(l_min,l_max, l_num)

train_r_squared = np.zeros(l_num)
test_r_squared = np.zeros(l_num)

pred_num = X.shape[1]
coeff_a = np.zeros((l_num, pred_num))
```


```python
from sklearn.model_selection import cross_val_score

for ind, i in enumerate(lambdas):    
    reg = Lasso(alpha = i)
    reg.fit(X_train, y_train)
    results = cross_val_score(reg, X, y, cv=5, scoring="r2")

    train_r_squared[ind] = reg.score(X_train, y_train)    
    test_r_squared[ind] = reg.score(X_test, y_test)
```


```python
# Plotting
plt.figure(figsize=(18, 8))
plt.plot(train_r_squared, 'bo-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
plt.plot(test_r_squared, 'bo-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)
plt.xlabel('Lamda value'); plt.ylabel(r'$R^2$')
plt.xlim(0, 19)
plt.title(r'Evaluate 5-fold cv with different lamdas')
plt.legend(loc='best')
plt.grid()
```


![](/post/2019-08-12-python-lasso-regression-auto/output_35_0.png)

### 1.8 Best Model

```python
df_lam = pd.DataFrame(test_r_squared*100, columns=['R_squared'])
df_lam['lambda'] = (lambdas)
# returns the index of the row where column has maximum value.
df_lam.loc[df_lam['R_squared'].idxmax()]
```




    R_squared    87.897525
    lambda        0.050000
    Name: 0, dtype: float64




```python
# Best Model
reg_best = Lasso(alpha = 0.144737)
reg_best.fit(X_train, y_train)
```




    Lasso(alpha=0.144737, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, reg_best.predict(X_test))
```




    3.635187490993961




```python
reg_best.coef_
```




    array([-0.34136411, -1.18223273, -0.        , -3.27132984,  0.        ,
            0.33262331,  0.71385488])



---
---
