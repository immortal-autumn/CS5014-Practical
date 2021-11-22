<center>CS5014 - Practical 2<br/>170025298	11/04/2021</center>

# Introduction
This practical aims to solve a specific problem about identifying the texture and colour of an object of the image with a more detailed model. The practical investigates a more precise model by using different algorithms. Validation of the model includes testing the balanced accuracy of the model and investigating the confusion matrix for easier identification of the model's problem.

EXT. During testing, I found that the position of the boarder does affect the accuracy of the model, so I added x and y property to the training. 


# Design
## Data Pre-processing

Pre-processing data is one of the most important stages of training the model. With data pre-processing, data that may lead to training inaccuracy or model segment fault will be removed, which keeps the model's stability. Data pre-processing stage does the following things:

- Clean missing data
- Clean error data
- Clean useless data
- Normalize data in the dataset
- Convert texture labels into numerical or binary (With one-hot encoding)

### Common pre-processing

Common pre-processing dataset do the data cleaning and data scaling. 

Since the dataset may contain useless data (For example, the image id and the object id), unexpected symbols (For example: want the column type is float, but get object actually), missing data (One blank is left Nan) or error values (For example, the width or height of the border is set 0, which is unable to recognize the object, but get labels for that), we may tackle with these data by fill correct data to the dataset, delete the row with missing data and correct the error data in the dataset.

These data value errors can be solved in the following steps:

- We can directly use the DROP function to drop the row or column of the training set with error values
- We can use the $fillNan()$ function to fill 0 to the empty area of the testing set (Whereas the set index should not be modified in case of uploading)

Scaling of the data decreases the hardness of scaling calculation and will decrement the effort of invalid data. Data are scaling scales the training and testing dataset separately. Scaling with training data will also record the mean value and expected value of the dataset and apply the scaling parameters to the testing set. Scaling of the test set uses the axiom as follows:
$$
X_f = (X - u) / S
$$
Whereas u is the mean value of the column and S is the standard value of the column. The result is shown as follows:

![image-20210414160532435](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210414160532435.png)

![image-20210414160556010](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210414160556010.png)

### Pre-processing General

![image-20210414160104474](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210414160104474.png)

Since the dataset has labels represents showing above, we would convert those labels numerically. We use the $get\_dummies()$ function retrieving all labels of the dataset and converting labels into numeric according to the label's position in the list of labels. Hence we can get the result shows as following:

![image-20210414160442788](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210414160442788.png) 

### Pre-processing with One-Hot encoding

For a linear model, if labels are converted into numeric, the training process may consider those numbers due to calculation from all other parts, which will lead to training confusion. As a result, we use one-hot encoding to encode texture and colour labels.

![image-20210414160646165](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210414160646165.png)

Importantly, labels that converted from the original label set should be stored to extract the correct output, as colour and texture are calculated with two different models. Using the $concat()$ function, we would append those results at the end of the dataset and remove the original colour and texture labels.

## Model Design

Model is the essential part of machine learning, as the precision of the model is being trained and evaluated in this stage. This section introduces a model that is considered to be suitable for solving this question. 

### Logistics Regression

Logistics regression is the model that is implemented with python (Using PyCharm). The chosen model considers the fact that this question is a question of classification. We would classify the image and judge the label (Colour, Texture) of the image. The logistics regression is a model for classification. For this model, hyper-parameters are chosen and with the reason lists as follows:

- multi-class (ovr or multinomial): As we have lots of labels in colour tag and texture tag, this is a multi-classification question. We choose ovr or ovo for classification. Multinomial uses multinomial regression for prediction, and ovr divides the question into several classifiers. (For example, we have [1,2,3] as outputs, classifier of over makes the probability represents as [1], [2, 3], which is one vs rest strategy)
- solver: The calculus of optimization problem. The property can be adjusted manually.
- C: C changes the generalization performance of the model. The higher the C, the more general the model represents, but the model would be less accurate.
- max_iteration: This property defines the maximum iteration of the model. It just does not need to be adjusted and can leave it default.
- penalty: The loss function type increment the availability and generalization performance of the model.
- class_weight:  The weight of classes of each model.

The model's hyperparameter is modified and adjusted to a best-match one, which could be the model that we want.

[IMPORTANT!] With data preparation, I annotated the represented line of data pre-processing. Actually, this model includes the process of data pre-processing, data validation and data training.

![image-20210416002112938](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210416002112938.png)

### Linear Regression

Linear Regression is one of the linear models but is being implemented with the Conda python environment (Jupiter Notebook). The model is a ordinary least squares Linear Regression, which is a classical model that solve the linear question. The model considers the image to be linear (May represents objects within the image looks obvious). Linear regression finds an axiom in the case of summarizing the motion of scatters. The calculus of the model may represent as follows:
$$
y = \beta_0 + \beta_1x_1 + ... + \beta_px_p
$$

Linear regression may considers the output to be numbers instead of labels, so we uses one-hot encoding to encode our output, and reverse our encoding in our final step. Since the output of the model would represents as a set of possibilities, we choose the highest possibility and set to one, and set the rest of labels to 0. 

Reverse from the one-hot encoding is implemented as follows:

```
[black, white] -> [[1, 0], [0, 1]]
if output = [1, 0]
label = labels[[[1, 0], [0, 1]].find(output)]
```



### Random Forest

Random Forest is one of the Ensemble learning models which is being implemented with the Conda python environment (Jupiter Notebook). Ensemble models represents its advantages with solving the problem of a single model may performs not quite well. According to the random forest documentation, the random forest creates a number of decision trees that any of the decision trees is unrelated to each other. Each decision tree will create a result representing the decision tree's output and the final classification of the prediction based on the number of decision tree voted to the label. Briefly, I consider the decision tree represents its calculus as follows:

- Train a decision tree with properties using the sampled dataset
- Choose a random property as splitting property of the decision tree
- Repeat the second step until any node of the decision tree is completely independent
- Repeat step 1-3 to create a variety of decision trees.

The random forest allows the following attributes being judged:

- n_estimator: This decides how many decision trees are generated.
- criterion: The evaluation of how well properties are being split with the decision tree.
- min_sample_split: This evaluates the minimum number of samples required to split an internal node.
- min_weight_fraction_leaf: This evaluates minimum weighted fraction of the sum total of weights required to be at a leaf node.
- min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
- oob_score: Use out-of-bag samples to increase the generalize property of the model.
- random_state: The randomness when building the tree.
- ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. 

A random forest calculates a variety of features and automatically allows for feature selection; moreover, the random forest balances the unbalanced dataset and keeps its accuracy when many features are missing.



## Model validation
### Accuracy identification


#### Accuracy Score

Accuracy score calculates the rate of the model returning true output. This tells about the distribution of responses. Calculus for accuracy score is described as follows:
$$
Accuracy = \frac{n_{correct}}{n_{total}}
$$


#### Balanced Accuracy Score

Balanced accuracy score is considered to solve the issue of imbalanced dataset is not evaluated precisely with accuracy score. Balanced accuracy score can be summarized as follows:
$$
Balanced = \frac{TPR + TNR}{2}
$$
 As TPR is the true positive rate and TNR is the true negative rate



#### Confusion Matrix

Confusion matrix evaluates the number of conditions that elements of the model is same as the predicted model. For example, if we predicts with [a, b, c, c] but we get [a, c, c, c] for true, we have the confusion matrix with

```
[
1,0,0
0,0,0
0,1,2
]
```

 


### K-Fold validation
We use k-Fold validation technology for testing the validation of the model. K-Fold technology splitting a dataset into several parts that have equal size. For each iteration, the model will use one part for validation and training with the rest of the model. The validation does helps to improve the precision rate of the model by modifying the hyper-parameter of the model. A preferred fold is to be considered $k=5$, that the model uses k-Fold technology that included in the sklearn package splitting the dataset into five parts.  The use of the k-fold validation represents as follows:
$$
Average\ Accuracy\ Score = \frac{u_1 + u_2 + u_3 + u_4 + u_5}{n}
$$

$$
Average\ Balanced\ Accuracy\ Score = \frac{A_1 + A_2 + A_3 + A_4 + A_5}{n}
$$

Whereas A represents the balanced accuracy score for each folded dataset, and u represents as the accuracy score for each folded dataset.


# Implementation

The implementation for this practical includes the following:

- All datasets that cleaned
- The implementation of the Linear Regression and represented result
- The implementation of the Logistics Regression model and represented result
- The implementation of the Random Forest model and represented result

# Testing

Validation test for each model represents as follows:

## Linear Regression

![Linear-Regression-none_para](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Linear-Regression-none_para.png)

As we can see above, Linear regression model represents some better than the logistics model, but the accuracy is still not satisfiable.

## Logistics Regression

![Logistics-Regression-C01-l2-ovo](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Logistics-Regression-C01-l2-ovo.png)

<center>This figure represents the accuracy of logistics regression with C=0.1, multiclass=Multinomial, penalty=l2 and solver=newton-cg.  </center>

![Logistics-Regression-C1-l2-ovo-max_50000](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Logistics-Regression-C1-l2-ovo-max_50000.png)

<center>This figure represents the accuracy of logistics regression with C=1, multiclass=Multinomial, penalty=l2 and solver=newton-cg. Max iteration is set 5000. </center>

![Logistics-Regression-C1-l2-ovr](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Logistics-Regression-C1-l2-ovr.png)

<center>This figure represents the accuracy of logistics regression with C=1, multiclass=ovr, penalty=l2 and solver=newton-cg.  </center>

As we can see above, changing parameters may represents less changes to the performance of the model. As a result, Logistics regression may represents as a less effective model for this practical.

## Random Forest

The figure showing below represents the performance of the model in validation stage. Random forest represents a higher accuracy than both linear regression model and logistics models, so the random forest might be a better model for the practical.

![Random_forest_n_estimate_120](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Random_forest_n_estimate_120.png)

## Final testing.

The image showing below represents the overall performance for each models, which looks less satisfiable:

![Testing_color](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Testing_color.png)

<center>Image for color recognition</center>

![Testing_texture](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Testing_texture.png)

<center>Image for texture recognition, reflected models represents as above accordingly</center>

Unfortunately, random forest represents not such satisfiable than it was done in the validation process, which may means that some features leads to overfitting of the model. We may need to identify which feature is strongly related to the model and may remove the feature in next step.

EXT. Accuracy for random forest have increased a bit. Image above does not represents the actual final result of the testing.


# Evaluation

## Linear regression and Logistics regression models

These two models performs not quite well in both validation step and testing step. This may caused by object identification is not a regression question, and hence is not able to be classified by Logistics Regression.

## Random Forest

When using k-fold strategy testing the performance of the model, 

![Random_forest_n_estimate_120](E:\BILI\P2Y2S1\CS5014-Practical\Practical2\report-data\Random_forest_n_estimate_120.png)

We can find out that Random tree performs well in validation stage, but is not generalized. I considers the issue might be leads by noises of pictures: Objects might be hidden or recognition of the object might be interrupted. 

As a result, the random forest represents overfit when predicting models with the test set. With analyzing the dataset, this might be caused by noises from images. As a result, the random forest performs not quite well with this dataset.


# Conclusion

As a result, random forest represents the best performance in object recognition, but causes overfitting in final testing. However, I believe that the random forest model represents a better performance with object identification in this practical. A further work for this practical is to do feature selection furtherly, as remove some highly related labels that may improve the performance of the model.

This practical let me learnt a lot. A further study of machine learning may focus on feature selection, as to find some more ways of identify features in the dataset and training with model with a better and more suitable dataset.