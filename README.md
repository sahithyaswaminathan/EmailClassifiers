# EmailClassifiers 

```
Sahithya Swaminathan
11.05.2018
```
## Prerequisites

Inorder to run this script, please install MATLAB version 2013 or above.

## Data Set

Make sure that the SpamDataset is available. You can check the below link for reference: [Spam Email Database](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.info.txt) 

## Project Scripts

### 1.Beta-Bernoulli Naive-Bayes Classifier

Beta-Bernoulli classifier has been compiled in two Matlab code files, one provides the accuracy and error rates on the test and train data
for alpha = 0 (hyper-parameter) and another file provides the accuracy and error rates for alpha values from 1 to 100 increamented by 0.5.

#### Run Command

* Beta - Bernoulli Naive Bayes classifier for alpha = 0

```
>> BETA_BERNOULLI_ALPHA_0 
```
* Beta - Bernoulli Naive Bayes classifier for alpha 1 to 100

```
>> BETA_BERNOULLI
```

#### Assumption: 

* Beta prior hyperparameters alpha and beta are assumed to be equal.

### 2. Gaussian Naive Bayes Classifier

Two Data normalization methods have been adopted:
* Log Normalization: Inputs are transformed using - log(x + 0.1)
* Z-Normalization: Each column is standardized to have 0 mean and unit variance. 

#### Run Command

Gaussian Naive Bayes for Log normalization

```
>> GAUSSIAN_LOG
```

Gaussian Naive Bayes for Z-Normalization

```
>> GAUSSIAN_ZNORM
```

### 3. Logistic Regression

Three data normalization method have been adopted:
* Log Normalization: Inputs are transformed using - log(x + 0.1)
* Z-Normalization: Each column is standardized to have 0 mean and unit variance. 
* Binarization: Input values greater than zero are normalized as 1 (x > 0)

#### Run Command

Logistic Regression for Log normalization

```
>> LOGISTIC_LOG
```

Logistic Regression for Z-Normalization

```
>> LOGISTIC_ZNORM
```

Logistic Regression for Binarization

```
>> LOGISTIC_BINARY
```
### 4.K- Nearest Neighbors 

Three data normalization method have been adopted:
* Log Normalization: Inputs are transformed using - log(x + 0.1)
* Z-Normalization: Each column is standardized to have 0 mean and unit variance. 
* Binarization: Input values greater than zero are normalized as 1 (x > 0)

Please make sure to have the function file (K_NN) on the same folder as that of the other KNN files

#### Run Command

KNN for Log normalization

```
>> KNN_LOG
```

KNN for Z-Normalization

```
>> KNN_ZNORM
```

KNN for Binarization

```
>> KNN_BINARY
```




