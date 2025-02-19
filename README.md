#Understanding the Data set#
From some early analysis, it can be concluded that the data set presents Perinatal Risks for
women, given their Age, Systolic/Diastolic Blood Pressure, Blood Sugar, Heart Rate and the
class of risk Type associated with each individual. The independent variables are composed out
of the predictors Age, Systolic/Diastolic Blood Pressure and Heart Rate. The dependent variable
is also sometimes called target, and it is represented by risk Type.
Classification is the appropriate methodology because of at least the following two fundamental
reasons. First, regression cannot represent a qualitative response (i.e. vs quantitative response)
with more than two classes. There is no natural way of representing variables that do not have a
natural ordering (e.g. Smartness, Wealth Type and Highschool Name) using the methods of
regression. The only case that regression might work would be for the binary case (e.g. Cancer
or No Cancer) that can be represented with 0 or 1 (i.e. indicating the absence/presence of the
mentioned variable). Second, regression cannot provide the correct probabilities of (Y|X) in the
range of [0, 1], as it might sometimes output values either negative responses or values outside
the specified range. Regression does not normalise its values in between [0, 1] and even if you
were to do that, the result will not be as expected (i.e. the outputs of the linear regression are
not probabilities).

#Data Exploration
I have explored the test and training sets using pair plots. Both are essentially coming from the
same data set, so all the distributions are somewhat similar. Besides the pair plots that show
similar distributions, I have also looked at the outliers from the training set using box plots. I
have not done the same for the test set because even if (most likely) there are outliers present
in the set, I will still not remove them / do anything about it because that will just overfit the data.
In the case of the train set, some of the data is skewed, and there are obvious outliers present.
The details of the way the data is distributed can be found in the notebook, as they are
accompanied by the visuals of the pair plots
The dataset is unbalanced, based on the fact that the classes do not have the same frequency.
The data is split “Low risk”: 40%, “Mid risk”: 32%, and “High risk”: 28%. This is based on the
amount of samples that have these classes.
For data pre-processing, I did a bunch of things to make the data ready for modelling. First, I
have checked if there are any null values within the training data, as that might prevent various
models from training on the data set. Second, I decided against dropping the duplicates from
the training set, which might overfit the model slightly. There are multiple philosophies to this,
but one can either leave the data in as duplicates, which assumes that those values are correct.
Otherwise, one can drop those values because they look like misreadings. In this case I decided
to not drop the duplicates, because if I was to then I will be not having enough examples to train
the systems (i.e. underfitting). Third, I used standardisation on the predictors because the
values did not have a common scale (i.e. how do you compare Blood Sugar with Body
Temperature on the sample scale)? This will essentially convert all the values into z-scores, that
allows all of them to be measured in terms of standard deviations. Fourth, I have decided to
replace one of the outliers with the mean from its column (i.e. it was an outlier for the Heart Rate
as 7, which looks like a misread). I have done this for the training set, but not for the test set,
because I might end up overfitting the data set by modifying it to my needs,
Algorithm Selection and Application

#Random Forest
Random Forests are a collection of Decision Trees. This is because the decision trees are weak
learners by themselves, but a random forest represents an "ensemble", meaning that together
with the other decision trees combined it becomes much more powerful (and with a lower
variance).
One important concept before introducing Random Forests, is that of Bagging or
Bootstrapping.Essentially Bootstrap aggregation, is a general procedure used to reduce the
variance of a statistical learning method.In a random forest, besides having Bootstrapped data,
we also select a random sample of predictors to be used for each split on each tree. The
algorithm is not allowed to consider a vast proportion of the predictors. This is because if we
were to only use bagged trees, then only the strongest predictors would be considered. The
overall outcome is a bunch of similar trees that are highly correlated. By only considering a few
predictors at a time, we decorrelate the trees.
This model is implemented through the sklearn library, and it has the following choice of
hyperparameters:
- N_estimators, which represents the number of trees inside the random forest. A higher
will usually be preferred as it will lead to more exploration (but there has to be a balance,
because it will be computationally costly to look for thousands of trees)
- Criterion, either using Gini Index or Entropy to calculate the Information Gain (i.e.
different calculation methods are easier to interpret, and/or sometimes one excels over
the other one in different contexts)
- N_jobs, which allows the usage of all the processors (i.e. speeds up our processing)
- Class weight, that allows us to describe to the model the way the data is distributed (i.e.
our data is unbalanced). This will help the model train on the data faster, as it already
has information on the distribution of data.
- Max_depth, that defines how deep our Decision Trees are going to be (i.e. prevents
overfitting). A deep tree might overfit the data, as it will try to keep splitting on various
predictors until it reaches the leaf nodes (but sometimes this is not efficient).

#Support Vector Machines
They have been popular since the 1990s, and dominated the field of Artificial Intelligence for a
while. It is also sometimes known as one of the best classifiers that does not require that much
tuning. Support Vector Machines (SVM) is essentially a generalisation of the maximal margin
classifier. However, this type of classifier cannot be used for most data sets because they must
be linearly separable. This issue is resolved by support vector classifiers, which overcomes the
problems of the maximal margin classifier, and allows it to be used for a range of cases.
Support Vector Classifiers classify data using optimal hyperplanes that maximise the margin
between the two closest points (each coming from one of the classes). Besides the margin,
there are support vector "lines" that pass through the mentioned points and determine the
maximal margin. Note, the importance of the margin is crucial, as this picks the best hyperplane.
SVM can be used for both linear and non-linear data, where the latter makes use of the kernel
trick to transform the data into higher dimensions (i.e. and this allows for linear separability).This
can happen by calculating the inner products (i.e. projecting using dot products) of the data
points, and projecting them into a higher dimension. The actual transformation does not happen,
but what matters is that it gives us the ability to separate the non-linear data in the higher
dimension. Calculating the margin can either result in a hard-margin or soft-margin
classification. In the case of hard-margin, we do not allow for misclassification, whereas
soft-margin allows for some misclassification. When considering the hyperparameters of SVM,
C is adjusting the margin (high C allows minimal misclassification, whereas a small C allows for
greater misclassification). As in all cases of hyperparameters, we just have to perform
trial-and-error to find the best option.
This model is also implemented through the sklearn library and uses the following
hyperparameters:
- Kernel, is used by the support vector machine to project the data into a higher
dimension. “Linear” simply moves the data from 1D to 2D, whereas “rbf” uses
infinite-dimensions for projections.
- Gamma tells us how much influence do two points have on each other (i.e. when using
the kernel trick, we compare two data points and then project them into
higher-dimension).
- C, is used to modify the soft margin of the support vector classifier. The higher, the more
misclassifications we allow between the two classes.
The choices of parameters for both models were made using the Grid Search, which is
essentially performing a combination operation on the parameters (one parameter keeps
changing while the others stay the same). It also uses cross-validation (10 times), that splits the
data into k folds (i.e. 10 times), and holds 1/k for testing on each fold. At the end of the cross
validation, we use the test set to check the actual accuracy of the model. This way, we test the
model k times, before we test it on the actual test data. The approach is applied to select the
best hyperparameters for the two models.

#Model Evaluation
The confusion matrix helps with seeing beyond “accuracy”, as if one was to only use that as a
metric, then it could possibly end up in misleading results. If we were to use a confusion matrix,
that uses True Positives (number of correctly predicted positive values), False Negatives
(number of predicted values as negatives, when the actual class is positive) and False Positives
(number of predicted values as positive, when the actual class is negative). Using these values,
in combination with the Recall out of the positive classes, how many were actually predicted
correctly) and Precision (out of the classes predicted positive, how many were actually positive),
one will be able to calculate the F-1 score (represents the overall accuracy of the model using
Recall and Precision). This value will be much more indicative of the overall accuracy of the
model, than simply checking how many values were correctly identified.
It is important to note that there is always a trade-off between Recall and Precision. High recall
means that the model tries to identify all the positives, but this will reduce the precision because
the model can incorrectly classify some positives as negatives. In contrast, low recall and high
precision means that the model is able to distinguish between negatives and positives, but it will
fail at capturing all the positives. This seems to be a recurring theme in Machine Learning, as a
similar parallel can be drawn with the bias-variance trade-off.
For the Random Forest model, I have achieved an overall F1 score of 0.8 (overall). This
indicates 80% “accuracy”. Contrary, the Support Vector Machine had a score of 0.76 (overall).
The reasons for the models being slightly different in terms of F1 scores is most likely due to not
enough hyperparameter exploration, small training data set, minimal preprocessing etc.. Other
than that, both models identify almost 80% of the test data correctly. However, if extra efforts
were to be made on this front, then one should be extremely careful not to overfit the data by
over selecting parameters that perform good on this data set, but poor on future ones (i.e. high
variance).
An important sign that shows our models are not underfitting, is that of the greater “accuracy” for
the training sets. If the training “accuracy” would have been lower than that of the test, then it
would have been a clear indication of underfit.
Besides simply analysing the confusion matrix one can have additional reasons for picking one
method over the other one. For instance, Decision Trees are usually easier to interpret
(depending on the size of the tree) and they might mirror the decision-making process of
humans. Therefore, after applying the Random Forest, we can visualise the chosen Decision
Tree much easier than the output of the Support Vector Machine (that brings data into higher
dimensions through projections). Another remark to be made is that both methods yielded
similar outputs, meaning that the either choice would have been a good one. The reason behind
this is because both are robust methods that work well in multi-class, multi-feature
environments.

#Ethics
Using Machine Learning in Health Care might have a lot of good and bad consequences.
Mainly, there will be concerns around how accurate the predictions of the models are. Also, one
must consider what decisions will be made based on the predictions. This includes treatments
and diagnosis. These are life-changing actions, and one must be extremely sure of the
performance of each model. Another issue that can bias the results is the data and the way it
was collected. Were there a lot of misreads? What region of the country were they recorded in?
Is the data a good representation of the population? One major concern is that of Automation
Bias, where decision-makers (e.g. doctors) will believe the outputs of AI/systems just because
they are “automatic”. This refers to blindly believing everything the system says, without
questioning decisions. An analogy would be the way a pilot will allow the auto-pilot to take
decisions without questions. This can lead to extremely bad decisions, where people’s lives will
be affected permanently. One must always consider the implications of using the machine
learning model, and the possible flaws within the system.

#Reference
James, G. et al. (2015) An introduction to statistical learning: With applications in R.
supervised learning scikit. Available at: https://scikit-learn.org/stable/supervised_learning.html
(Accessed: 20 October 2024).
Russell, S.J. and Norvig, P. (2009) Artificial Intelligence. Upper Saddle River, N.J: Pearson
Education
