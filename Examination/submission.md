# CS5014 Exam

## Question 1

#### Question a

##### Question 1

Loss terms:
$$
\sum_{i=1}^m(y^i-\theta_0-\theta^Tx^i)^2
$$
Regularisation terms:
$$
\lambda_1\sum_{j=1}^P|\theta_j|\\\lambda_2\sum^p_{j=1}|\theta_j^2|
$$

##### Question 2

$$
f(x) = \sum_{i=1}^{m}\theta^Tx^i + \theta_0
$$

Which can be written as
$$
f(x) = \sum_{i=1}^{m}\theta^Tx^i + \theta_0 \times 1 = [\theta_0, \theta_1, ..., \theta_n][\begin{matrix}1\\x_1\\x_2\\...\\x_n\end{matrix}] = \theta_0 + \theta_1x_1 + ... + \theta_nx_n
$$
Hence this is obviously a linear model.



##### Question 3

Regression model. As is shown in question 2, the property is numerical, and this model is a linear regression model that can also be formed as:
$$
y^{(i)} = f(x^{(i)}; \theta)
$$
Hence the model is a regression model.

##### Question 4

Hyperparameter: $\theta_0, \theta_j$

Parameter: $\lambda_1, \lambda_2$

##### Question 5

Assume we have a training set and a testing set, we can use k-Fold to divide the training set into training and validation set, and use cross validation method testing the the model with the best performance, and hence decide the hyperparameter and parameter of the model. Finally, we will use the best parameter of the model to predict on the testing set.



#### Question b

##### Question 1

Disagree. Accuracy evaluates the percentage of true result in the whole testing set, however the accuracy is unable to evaluate the performance of the model when the positive and negative output is severely unbalanced. Consider the condition that a training set with 98% positive and 2% negative, if the model simply take all inputs and return positive, the accuracy is incredibly 98%. However, the model is unable to solve the problem.

##### Question 2

Disagree. A hidden layer allows for linear calculation, if we already have a linear model $g(x)$, with the property of linear 
$$
f(g(x^{(i)}; \theta_0), \theta_1) = \theta^T_1(\theta_2^Tx^{(i)})^i
$$
And the model is still a linear model when all activation function is linear as well.

##### Question 3

Disagree. 

Newtons' Method can represents like a wave jumping through the convergent point. The method is represents as:
$$
\theta_{t+1} \leftarrow \theta_t - H_t^{-1}g_t
$$
as $H_t^{-1}g_t$ is the direction of moving with newton's method. The function move according to the derivation of the model and hence not able to converge in a single iteration.

##### Question 4

Disagree. Valuable feature should represents with the highest variance. Larger eigen-values means larger variance in the projection. Choosing two smallest eigenvalues is unable to retain as much variance as possible.

##### Question 5

Disagree. We want the minimum of $L$, which means that the question requires that $\beta_0 = 0$. However, the red line shows as a horizontal line, which makes it impossible that $\beta$ is set 0. Hence the red line could not represents the fitted model.

## Question 2

#### Question a

##### Question 1

We would make that the blue dot to be positive and yellow triangle to be negative.

Accuracy:
$$
Accurate = \frac{TP + TN}{P + N} = \frac{9}{11} \approx 0.8181
$$


Precision:
$$
Precision = \frac{TP}{TP+FP} = \frac{5}{6} \approx 0.8333 
$$


Recall:
$$
Recall = \frac{TP}{TP + FN} = \frac{5}{6} \approx 0.8333
$$


F1-Score:
$$
F1-Score = \frac{2 * (Precision * Recall)}{Precision + Recall} = \frac{\frac{50}{36}}{\frac{10}{6}} = \frac{5}{6}
$$


##### Question 2

Since the dataset is not linearly separable, this is a soft-margin solution.

The SVM uses polynomial kernel, as the graph is the representation of a line that is linear, where the only kernel that satisfies the condition is the polynomial kernel.

##### Question 3

Support Vector: (C is the Lagrange multipliers)
$$
\left
\{
\begin{matrix}
\alpha\in[0, C]&:&Point[8, 4]\\
\alpha=C&:&Point[5, 6]
\end{matrix}
\right.
$$
Slack Variables:  
$$
\{
5: \epsilon_1; 6:\epsilon_2
\}
$$


##### Question 4

Decision boundary and margin does only affected by a small number of points, which are called support vectors. Removing point 10 will not affect the support vector of the model, and hence will leads no effect to both decision boundary and margin.

#### Question b

##### Question 1

The kernel trick mapping patterns to a high-dimensional feature space F, as the dot product is used for comparing them.

If we have data $x, x'\in X$ and a mapper $h:X\to\R^N$, then the kernel function is:
$$
k(x, x') = <h(x), h(x')>
$$
Where the feature space is being chosen that allows for the dot product that can be directly being evaluated with such function in the input space. This does avoid working in space F.

##### Question 2

Point [4, 5, 6, 8] are training points that are needed to classify a new data point x. As is described in the question, classification equation is given by:

![image-20210515113059644](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210515113059644.png)

As $x^i$ is associated with none-zero $\alpha_i$. None zero support vector have points [4,5,6,8], which represents the answer.

#### Question c

##### Question 1

The polynomial maps from low dimension to high dimension, which can be represents as $<h(x), h(x')>$. $k(x) = (1+<x, x'>)^3$ represents as a polynomial inner product kernel, which can be formed as $K(x, x') = (1+<x, x'>)^d = <h(x), h(x')>$. Since the dimension of the input feature $[x_1, x_2]^T$ is 2, so we have $d=n(n+1)/2 = 3$, which means that the dot product will working on using vectors in a space of dimension 3, which means that k(x) is a valid polynomial kernel.

##### Question 2

$R=12$
The expanded feature of R represents as a form of combination of the dot product of $<h(X), h(X')>$


##### Question 3

The classification of the SVM represents as:
$$
f(x)=\theta_0+\sum_{i=1}^N\hat{a_i}y_iK(x, x_i)
$$
And the linear model predict on new data as:
$$
f(x;\theta)=\theta^Tx^{(i)}
$$
where $x_0$ is assumed to be 1.

Since predicting on linear kernel, inputs are mapping to high dimension, which cost much more time than using the SVM with polynomial kernel, SVM performs better than linear kernel is this case.





## Question 3

#### Question a

Both K-means and EM are clustering algorithm, and K-means can be regarded as a specific case of EM algorithm for the specific Gaussian mixture.

The expectation step E:
$$
w_{ik}\leftarrow\left\{
\begin{matrix}
1, &k=argmax_{k'}p(z^i=k'|x^i)\\
0, &otherwise
\end{matrix}
\right.
$$
The Maximisation step M:

With priors $\pi_1, \pi_2, ... \pi_k$ assumed to be same and $\sum_1 = \sum_2=...= \sum_k=I$

Hence the assignment step is as follows:
$$
k=argmax_{k'}p(z^i=k'|x^i) = argmax_{k'}min||x^i-\mu_{k'}||_2^2
$$
Then the update step:
$$
\mu_k \leftarrow \frac{\sum^m_{i=1}I(z^i=k)x^{(i)}}{\sum^m_{i=1}I(z^i=k)} 
= \frac{\sum_{i=1}^mw_{ik}x^{(i)}}{\sum_{i=1}^{m}w_{ik}}
$$
Which proves the view point.

#### Question b

EM outperformed on boundary checking and convincing. K-means is observed to have limitations on checking boundary points, and The model uniform distance measure to all clusters. But this limitation is being solved by EM. Moreover, K-means do not have a underlying model that supports it. The solution of K-means tends to be more intuition based, which is less convincible. As a result, EM performs better than K-means where the shape of clusters are strange or complicated.

K-Means performs better on detecting homogeneous cluster. EM repeating expectation and maximisation steps, which tends to be more complex than K-means; At the same time, since assignment of K-means is hard coded, this save time on expectation step. 



#### Question c

Disagree. Larger K may cause overfitting.

We can pick K use Bayesian Information Criteria:
$$
BIC(M_i) = logP(X|\hat{\phi}_{ML}, M_i) - \frac{d_i}{2}log(m)
$$
where i represents the model i, which is the range of K for mixture models. In support for the BIC, models are penalized when uses large K,  we can find the largest BIC to decide the correct value of K to prevent from overfitting.

#### Question d

The expectation step for Gaussian Mixture is
$$
for
\left\{ 
\begin{matrix}
i\in[1, m]\\k\in[1, K] 
\end{matrix}
\right.
w_{ik}\leftarrow p(z^i=k|x^i) = \frac{\pi_kN(x^i; \mu_k, \sum_k)}{\sum_{j=1}^K\pi+jN(x^i; \mu_j, \sum_j)}
$$
The maximisation step M is:
$$
for\ k\in[1,K]\qquad 
\left\{
\begin{matrix}
\pi_k&\leftarrow&\frac{1}{m}\sum_1^mw_{ik}\\
\mu_k&\leftarrow&\frac{1}{\sum_1^mw_{ik}}\sum_{i=1}^{m}w_{ik}x^{(i)}\\
\sum_k&\leftarrow&\frac{1}{\sum_1^mw_{ik}}w_{ik}(x^{(i)}-\mu_k)(x^{(i)}-\mu_k)^T
\end{matrix}
\right.
$$

We assume the following assumptions:

- Prior distribution of the membership $z^i$ is the same across the K components (pi is already given)
- A m*n dataset X is already given
- K represents as the cluster group that is pre-defined
- A function repmat(matrix, i, n) performs the same function as MATLAB
- Operands \' (transpose), .^, ./ represents the same function as MATLAB
- Clustering is done randomly (initialization) in the program before EM steps.

```Python
def guassian_calculation(x, mu, sigma):
	m = mu.shape[1]
	inn_res = sum(sum(sigma.transpose() * sigma))
	return 1./((2 * pi).^(m/2) * inn_res.^(1/2)) * exp(-0.5 * (x-mu) * signma.inverse() * ((x-mu).transpose()))
```

```Python
# Calculate weights for all data point of X
def e_step(X, K, phi, mu, sigma):
	m, n = X.shape
    ret = np.zeros(m, K)
	# Calculate the density of probability for it in jth clustering group                                
	for i in range(0, m)     
    	P = np.zeros(m, K)
        for j in range(0, K):
        	P[i, j] = phi[j, 0] * guassian_calculation(X[i, :], mu[j, :], sigma[:, :, j])
        P_sum = sum(P[i, :])
        for j in range(0, K):
            ret[i, j] = P[i, j] / P_sum
    # weight for N * K matrix of points in X
 	return ret
```


```Python
# Updating and maximize the parameter phi, u, sigma
def m_step(X, zs):
    mu = np.array()
    # number and dimension of the training set
    m, n = X.shape
    K = zs.shape[1]
    zs_k = sum(zs[0])
    phi = (zs_k / m).transpose()
    for i in range(K):
        mu[i, :] = sum(repmat(zs[:, 0], 1, n) .* X, 1) ./ repmat(zs_k[i], 1, n)
    for i in range(K):
        sigma[:, :, i] = (repmat(zs[:, 0], 1, n) .* X).transpose() * X ./ repmat(zs_k[i], n, n) - mu[i, :].transpose() * mu[i, :]
    return phi, mu, signma

```



#### Question e

Since the general mixture model is 
$$
p(x)=\sum_{k=1}^K\pi_kf(x; \theta_k)
$$
Following with the template, we now have the E step:
$$
w_{ik}\leftarrow\frac{\pi_kf(y)}{\sum_{j=1}^K\pi_j(f(y))}
$$
The M step for $\pi$ is available based on the template and the definition of $w_{ik}$
$$
\pi_k \leftarrow \frac{1}{m}\sum_1^mw_{ik}
$$

$$
\theta \leftarrow argmax_\theta\sum^m_{i=1}w_{ik}logf(y)
$$

We would make the following assumptions:

- Prior distribution of the membership $z^i$ is the same across the K components (pi is already given)
- A m*n dataset X is already given
- K represents as the cluster group that is pre-defined
- A function repmat(matrix, i, n) performs the same function as MATLAB
- Operands \' (transpose), .^, ./ represents the same function as MATLAB
- Clustering is done randomly (initialization) in the program before EM steps.

Where the weighted likelihood for f is already given by the question. Hence we can get the following:

```Python
def calculator(X, u, c):
    // density calculator
	return c * exp(u.transpose() * x)
```

```Python
def e_step(X, K, c, u, pi):
    m, n = X.shape
    ret = np.zeros(m, K)
    for i in range(0, m)     
    	P = np.zeros(m, K)
        for j in range(0, K):
            P[i, j] = phi[j, 0] * calculator(X[i, :], u[j, :], c)
        P_sum = sum(P[i, :])
        for j in range(0, K):
            ret[i, j] = P[i, j] / P_sum
 	return ret
```

```python
def m_step(X, zs):
    u = np.array()
    m, n = X.shape
    K = zs.shape[1]
    zs_k = sum(zs[0])
    phi = (zs_k / m).transpose()
    for i in range(K):
        u[i, :] = sum(repmat(zs[:, 0], 1, n) .* X, 1) ./ repmat(zs_k[i], 1, n)
    return phi, u
```





