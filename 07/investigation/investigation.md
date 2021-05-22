# Investigation of binary classificator
Lukas Forst

First we need to prepare some usefull python code and do some data exploration to see what is going on.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score,roc_auc_score

np.random.seed(123)

%matplotlib inline
```


```python
data_root = 'data'
ground_truth_file = 'GT'
c_names = np.array(['C1', 'C2', 'C3', 'C4', 'C5'])
# data X[j]
j = 100
# number of alpha parameters
k = 50
# number of classificators
i = 5
```


```python
c_data = np.array([
    np.genfromtxt(f'{data_root}/{c}.dsv', delimiter=',',dtype=np.dtype('uint8'), encoding='utf-8')
    for c in c_names
])
c_data.shape
```




    (5, 100, 50)




```python
y = np.genfromtxt(f'{data_root}/{ground_truth_file}.dsv',dtype=np.dtype('uint8'), encoding='utf-8')
y.shape
```




    (100,)




```python
def binary_classification_performance(data, y_true, c, c_names, a):
    y_pred = data[c, :, a]
    
    tn, fp, fn, tp = confusion_matrix(y_true = y_true, y_pred = y_pred).ravel()
    accuracy = accuracy_score(y_true = y_true, y_pred = y_pred)
    precision = precision_score(y_true = y_true, y_pred = y_pred, zero_division = 0)
    recall = recall_score(y_true = y_true, y_pred = y_pred, zero_division = 0)
    f1_score = (2 * precision * recall / (precision + recall)) if precision + recall != 0 else math.nan

    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    auc_roc = roc_auc_score(y_true = y_true, y_score = y_pred)

    result = pd.DataFrame({
        'Alpha': [a],
        'Classificator': [c_names[c]],
        'Accuracy' : [accuracy],
        'Precision/PPV' : [precision],
        'Recall/Senitivity/TPR' : [recall],
        'F1 Score' : [f1_score],
        'AUC_ROC' : [auc_roc],
        'Specificty/TNR': [specificity],
        'NPV' : [npv],
        'True Positive' : [tp],
        'True Negative' : [tn],
        'False Positive':[fp],
        'False Negative':[fn]})
    return result
```


```python
pds = [binary_classification_performance(c_data, y, c, c_names, a) for a in range(k) for c in range(i)]
df = pd.concat(pds)
df = df.set_index(['Classificator', 'Alpha'], inplace=False)
df.head()
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
      <th></th>
      <th>Accuracy</th>
      <th>Precision/PPV</th>
      <th>Recall/Senitivity/TPR</th>
      <th>F1 Score</th>
      <th>AUC_ROC</th>
      <th>Specificty/TNR</th>
      <th>NPV</th>
      <th>True Positive</th>
      <th>True Negative</th>
      <th>False Positive</th>
      <th>False Negative</th>
    </tr>
    <tr>
      <th>Classificator</th>
      <th>Alpha</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C1</th>
      <th>0</th>
      <td>0.92</td>
      <td>0.862069</td>
      <td>1.0</td>
      <td>0.925926</td>
      <td>0.92</td>
      <td>0.84</td>
      <td>1.0</td>
      <td>50</td>
      <td>42</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C2</th>
      <th>0</th>
      <td>0.64</td>
      <td>0.581395</td>
      <td>1.0</td>
      <td>0.735294</td>
      <td>0.64</td>
      <td>0.28</td>
      <td>1.0</td>
      <td>50</td>
      <td>14</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C3</th>
      <th>0</th>
      <td>0.52</td>
      <td>0.510204</td>
      <td>1.0</td>
      <td>0.675676</td>
      <td>0.52</td>
      <td>0.04</td>
      <td>1.0</td>
      <td>50</td>
      <td>2</td>
      <td>48</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C4</th>
      <th>0</th>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.50</td>
      <td>1.00</td>
      <td>0.5</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>C5</th>
      <th>0</th>
      <td>0.63</td>
      <td>0.574713</td>
      <td>1.0</td>
      <td>0.729927</td>
      <td>0.63</td>
      <td>0.26</td>
      <td>1.0</td>
      <td>50</td>
      <td>13</td>
      <td>37</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Selection of optimal parameter
We're looking for an optimal alpha parameter for classifier "C1" without context.


```python
# select data just for a C1
c1 = df.loc['C1']
```

### Accuracy
We're now looking for as many correct classifications as possible and we don't have prefference whether False Negative or False Positive is better or worse


```python
c1[c1['Accuracy'] == c1['Accuracy'].max()]
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
      <th>Accuracy</th>
      <th>Precision/PPV</th>
      <th>Recall/Senitivity/TPR</th>
      <th>F1 Score</th>
      <th>AUC_ROC</th>
      <th>Specificty/TNR</th>
      <th>NPV</th>
      <th>True Positive</th>
      <th>True Negative</th>
      <th>False Positive</th>
      <th>False Negative</th>
    </tr>
    <tr>
      <th>Alpha</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>0.97</td>
      <td>0.979592</td>
      <td>0.96</td>
      <td>0.969697</td>
      <td>0.97</td>
      <td>0.98</td>
      <td>0.960784</td>
      <td>48</td>
      <td>49</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.97</td>
      <td>0.979592</td>
      <td>0.96</td>
      <td>0.969697</td>
      <td>0.97</td>
      <td>0.98</td>
      <td>0.960784</td>
      <td>48</td>
      <td>49</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.97</td>
      <td>0.979592</td>
      <td>0.96</td>
      <td>0.969697</td>
      <td>0.97</td>
      <td>0.98</td>
      <td>0.960784</td>
      <td>48</td>
      <td>49</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Here we can see that the **best accuracy** has **alpha = [22, 23, 24]** that actually have the same properties. So if we were to optimize on accuracy, we would select these parameters.

### Sensitivity
On the other hand, sometimes we don't want to have false negatives and we want to be as sensitive as possible (and rather mark something as False Positive, instead of creating False Negatives), so we optimize on sensitivity.


```python
c1[c1['Recall/Senitivity/TPR'] == c1['Recall/Senitivity/TPR'].max()]
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
      <th>Accuracy</th>
      <th>Precision/PPV</th>
      <th>Recall/Senitivity/TPR</th>
      <th>F1 Score</th>
      <th>AUC_ROC</th>
      <th>Specificty/TNR</th>
      <th>NPV</th>
      <th>True Positive</th>
      <th>True Negative</th>
      <th>False Positive</th>
      <th>False Negative</th>
    </tr>
    <tr>
      <th>Alpha</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.92</td>
      <td>0.862069</td>
      <td>1.0</td>
      <td>0.925926</td>
      <td>0.92</td>
      <td>0.84</td>
      <td>1.0</td>
      <td>50</td>
      <td>42</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.93</td>
      <td>0.877193</td>
      <td>1.0</td>
      <td>0.934579</td>
      <td>0.93</td>
      <td>0.86</td>
      <td>1.0</td>
      <td>50</td>
      <td>43</td>
      <td>7</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Here we can see that the there're two alphas, that have sensitivity 1, meaning that they marked all positive values as positive. However, the alpha = 1 is a bit better, as it has better accuracy, so if we were to optimize for **sensitivity**, we would choose **alpha = 1**.

### Specificity


```python
tnrs = c1[c1['Specificty/TNR'] == c1['Specificty/TNR'].max()]
print(f'Best Alphas according to True Negative Rate: {len(tnrs)}')
```

    Best Alphas according to True Negative Rate: 12


In terms of True Negative Rate metric, there're many good alpha parameters. Thus, we need to select the best parameter using Accuracy, or any other metric - in our case, all metrics give us the same winner:


```python
tnrs[tnrs['Accuracy'] == tnrs['Accuracy'].max()]
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
      <th>Accuracy</th>
      <th>Precision/PPV</th>
      <th>Recall/Senitivity/TPR</th>
      <th>F1 Score</th>
      <th>AUC_ROC</th>
      <th>Specificty/TNR</th>
      <th>NPV</th>
      <th>True Positive</th>
      <th>True Negative</th>
      <th>False Positive</th>
      <th>False Negative</th>
    </tr>
    <tr>
      <th>Alpha</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>0.63</td>
      <td>1.0</td>
      <td>0.26</td>
      <td>0.412698</td>
      <td>0.63</td>
      <td>1.0</td>
      <td>0.574713</td>
      <td>13</td>
      <td>50</td>
      <td>0</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



The best alpha parameter when we optimize on **Specificity** (or True Negative Rate) is **alpha = 38**.

### Visualisation

Let's visualise that on the plot:


```python
c1sb = c1.plot(
    y=['Accuracy', 'Precision/PPV', 'Recall/Senitivity/TPR', 'Specificty/TNR'], 
    figsize=(16,10),
    ylabel='Value',
    xlabel='Alpha',
    title='C1'
)
# accuracy, alpha = 22,23,24, value =c1['Accuracy'].max()
c1sb.plot([22,23,24], [c1['Accuracy'].max()]*3, linewidth=5, label='Best Accuracy')
# sensitivity, alpha = 1, value c1['Recall/Senitivity/TPR'].max()
c1sb.scatter(1, c1['Recall/Senitivity/TPR'].max(), s=100, label='Best Sensitivity')
# specificity, alpha = 38, value c1['Specificty/TNR'].max()
c1sb.scatter(38, c1['Specificty/TNR'].max(), s = 100, label='Best Specificity')
l = c1sb.legend()
```


    
![png](output_22_0.png)
    


## Sum up
We've found five "best" values for parameter *alpha = [1, 22, 23, 24, 38]*. There's no single best solution, it depends on which metric we want to optimize - either *Accuracy* when we want to have as many correct classifications as possible and we don't prefer FP/FN, or *Sensitivity* where we don't want to miss any positive classifications but we're OK with some false positives, or *Specificity* where we're minizing false positives.

## Top Secret
The aim is to find classifier that won't allow anybody else to open a safe with our fingerprint. In this case we don't want to allow False Positives (so no foreign fingerprint is marked as valid).


```python
# solution for the most secure thingy
no_fp = df[df['False Positive'] == df['False Positive'].min()]
no_fp[no_fp['Accuracy'] == no_fp['Accuracy'].max()]
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
      <th></th>
      <th>Accuracy</th>
      <th>Precision/PPV</th>
      <th>Recall/Senitivity/TPR</th>
      <th>F1 Score</th>
      <th>AUC_ROC</th>
      <th>Specificty/TNR</th>
      <th>NPV</th>
      <th>True Positive</th>
      <th>True Negative</th>
      <th>False Positive</th>
      <th>False Negative</th>
    </tr>
    <tr>
      <th>Classificator</th>
      <th>Alpha</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C4</th>
      <th>11</th>
      <td>0.73</td>
      <td>1.0</td>
      <td>0.46</td>
      <td>0.630137</td>
      <td>0.73</td>
      <td>1.0</td>
      <td>0.649351</td>
      <td>23</td>
      <td>50</td>
      <td>0</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the the **best classifier**, that does not have False Positives (thus won't allow intruder to unlock the box with documents) and has the best precision is the **C4** with **alpha = 11**.

## Safety first
The aim here is to create a function that would return True if given classifier is better or worse then the one selected in the previous section.


```python
# select previous clasifier
c4_a11 = df.loc['C4', 11]
necessary_fp = c4_a11['False Positive']
necessary_accuracy = c4_a11['Accuracy']

print('In order to be better then C4 with alpha=11, classifier needs to achieve:')
print(f'False Positives: <= {necessary_fp}')
print(f'Accuracy: >= {necessary_accuracy}')
```

    In order to be better then C4 with alpha=11, classifier needs to achieve:
    False Positives: <= 0.0
    Accuracy: >= 0.73



```python
def is_better_then_c4_a11(c6):
    def create_pd(alpha):
        y_pred = c6[:, alpha]
        y_true = y
        tn, fp, fn, tp = confusion_matrix(y_true = y_true, y_pred = y_pred).ravel()
        accuracy = accuracy_score(y_true = y_true, y_pred = y_pred)
        return pd.DataFrame({'False Positive': [fp], 'Accuracy': [accuracy]})

    pds = [create_pd(a) for a in range(k)]
    df = pd.concat(pds)

    fps = df[df['False Positive'] == df['False Positive'].min()]
    # when c6 has less false positives then the c4_a11 then it is better right away
    if fps['False Positive'].min() < necessary_fp:
        return True
    # when it has same number of false positives but it has better accuracy, then it is better
    elif fps['False Positive'].min() == necessary_fp and fps['Accuracy'].max() > necessary_accuracy:
        return True
    # otherwise it is worse
    else:
        return False
```

Let's try this function - in theory all classifiers should return false (if we selected in the previous step the best one.


```python
better = np.count_nonzero(np.array([is_better_then_c4_a11(c_data[c, :, :]) for c in range(i)]))
print(f'Classifiers better then the C4 alpha = 11: {better}')
```

    Classifiers better then the C4 alpha = 11: 0


This function seems to be correct and the previous step as well as the function didn't mark any classifier as better then the C4 alpha 11.
