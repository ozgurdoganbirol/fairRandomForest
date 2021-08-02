# fairRandomForest
I have experimented to come up with a method to debias a [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) by tree weighing. I work with with the [bank marketing dataset from UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

## Method

A random forest model trained for binary classification problem has many trees in the ensemble. The majority vote of trees might result with biased labels against a sensitive variable (race, gender). However, since random forest bootstraps the data and picks random features in each tree, the degree of bias of the decisions of the single trees will be different than of the ensemble.

## Fairness Criterion

The fairness notion of [equality of opportunity](https://ai.googleblog.com/2016/10/equality-of-opportunity-in-machine.html) is aimed in the experiment. For many real life problems, the desired quality of fairness is parallel with this criterion. The following function is used to evaluate the difference of equality of opportunity between the values of discriminative variable (disc) where 0 is the disadvantaged class (for example, women), and 1 is the advantaged.

```python
def diff_eoo(y,pred,disc): #Function to calculate difference of equality of opportunity
    data=list(zip(y,pred,disc))
    data=pd.DataFrame(data,columns=["y","pred","disc"])
    #Opportunity achievement rate of advantaged and disadvantaged individuals
    opp_adv=data[(data.disc==1) & (data.y==1) & (data.pred==1)].shape[0]/data[(data.disc==1) & (data.y==1)].shape[0]
    opp_dis=data[(data.disc==0) & (data.y==1) & (data.pred==1)].shape[0]/data[(data.disc==0) & (data.y==1)].shape[0]
    return round(opp_adv-opp_dis,2) #Difference of equality of opportunity
```
