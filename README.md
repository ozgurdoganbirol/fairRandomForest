# fairRandomForest
The debiased model of a random forest by tree weighing. A random forest model trained for binary classification problem has many trees in the ensemble. The majority vote of trees might result with biased labels against a sensitive variable (race, gender). However, since random forest bootstraps the data and picks random features in each tree, the degree of bias of the decisions of the single trees will be different than of the ensemble.

In this practice, I work with the popular banking dataset from UCI machine learning library to show that the fairness notion of equality of opportunity could be sustained through weighing of the decisions of the trees in a random forest ensemble proportionally inversed by their amount of bias.
