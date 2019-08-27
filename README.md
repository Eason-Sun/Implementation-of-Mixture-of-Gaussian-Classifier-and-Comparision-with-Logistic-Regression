# Implementation-of-Mixture-of-Gaussian-Classifier-and-Comparision-with-Logistic-Regression

The first part of this project implements the Mixture of Gaussian classifier from scratch (without using any existing machine learning libraries e.g. sklearn) for optical recognition of handwritten digits dataset. 

The second part of this project focuses on the analysis in terms of classification accuracy, time efficiency and expressivity.

My implementation of Logistic Regression: https://github.com/Eason-Sun/Implementation-of-Logistic-Regression-Using-Newtons-Method
## Dataset:
Link: http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits  

The data used for this project is a modified version of the Optical Recognition of Handwritten Digits Dataset from the UCI repository. 
It contains pre-processed black and white images of the digits 5 and 6. Each attribute indicates how many pixels are black in a patch of 4 x 4 pixels.

### Format: 
There is one row per image and one column per attribute. The class labels are 5 and 6. The training set is already divided into 10 subsets for 10-fold cross validation.

## Parameter Estimation
𝜋 (prior class probability),   
𝝁𝟏, 𝝁𝟐 (mean of classes),  
𝚺 (Covariance matrix) are estimated using Maximum Likelihood Estimation (MLE).

## Classification Accuracy Comparision:

![Capture](https://user-images.githubusercontent.com/29167705/63806769-537b4200-c8ea-11e9-9f8a-7bc872ba7c48.JPG)

## Time Efficiency Comparision:

Mixture of Gaussians assumes that data points of both classes are drawn from some Gaussian distributions. It uses Maximum Likelihood Estimation (MLE) to estimate the parameters of two Gaussians as well as the prior so that It can apply Maximum A Posteriori (MAP) to determine the class label which yields the maximum posterior between two classes.

Logistic regression assumes that data points of both classes are drawn from some distributions of the exponential family. For binary classification, the general form of posterior is the sigmoid function whose input is a linear relation of data points X. Negative Log likelihood is adopted as a loss function to find such linear relationship w that is optimal. However, it’s difficult to obtain a closed form solution for w by setting the gradient to 0. Therefore, iterative methods such as gradient descent and Newton’s method are the ways to go. Newton’s method is more preferable for faster convergency, but it’s still much slower than Mixture of Gaussians.


![Capture](https://user-images.githubusercontent.com/29167705/63806858-83c2e080-c8ea-11e9-9356-f75293a012ea.JPG)

## Expressivity Discussion (Linear vs Nonlinear):
Generally, the linear separators are less expressive so that it performs stably, which leads low variance, but might suffer from high bias. Linear separators are well suited for linearly separable data.

On the other hand, non-linear separators such as KNN have more expressive power and could differentiate arbitrary data very well, which ensures low bias. However, it becomes more data dependent and sensitive to noise, in other words, high variance. Non-linear separators are best performant when dealing with non-linear pattern, i.e. manifold classification.
KNN has ~80% accuracy in cross-validation, but it noticeably degrades in classification for testing data (~73%). In Logistic Regression, testing set even has a higher score than cross-validation set in training phase (~89% vs ~87%). Therefore, for this dataset, linear models generalize better.

