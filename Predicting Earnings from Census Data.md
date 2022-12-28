Predicting Earning from Census Data
================
2022-12-22

### About the Study and the Data Set

The US government regularly gathers data on demographics through a
census. In this study, we will use information from the census to
predict an individual’s yearly income, specifically whether they earn
more than \$50,000. The data comes from the UCI Machine Learning
Repository and is stored in the file census.csv. It includes information
on 31,978 people in the US.  
The dataset includes 13 variables: age, workclass, education,
maritalstatus, occupation, relationship, race, sex, capitalgain,
capitalloss, hoursperweek, nativecountry, and over50k. The age of the
individual is given in years, and the workclass describes the
individual’s employment status (e.g., whether they work for the
government or are self-employed). The education variable indicates the
highest level of education the individual has completed. Maritalstatus
indicates the individual’s marital status, and occupation describes the
type of work they do. The relationship variable specifies the
individual’s relationship to their household, and race and sex provide
the individual’s racial and gender identities. The capitalgain and
capitalloss variables show the amount of money the individual gained or
lost through the sale of assets in 1994. The hoursperweek variable
indicates how many hours per week the individual works, and
nativecountry specifies the individual’s country of origin. Finally, the
over50k variable indicates whether the individual earned more than
\$50,000 in 1994.

### EDA

``` r
census = read.csv("census.csv")
str(census)
```

    ## 'data.frame':    31978 obs. of  13 variables:
    ##  $ age          : int  39 50 38 53 28 37 49 52 31 42 ...
    ##  $ workclass    : chr  " State-gov" " Self-emp-not-inc" " Private" " Private" ...
    ##  $ education    : chr  " Bachelors" " Bachelors" " HS-grad" " 11th" ...
    ##  $ maritalstatus: chr  " Never-married" " Married-civ-spouse" " Divorced" " Married-civ-spouse" ...
    ##  $ occupation   : chr  " Adm-clerical" " Exec-managerial" " Handlers-cleaners" " Handlers-cleaners" ...
    ##  $ relationship : chr  " Not-in-family" " Husband" " Not-in-family" " Husband" ...
    ##  $ race         : chr  " White" " White" " White" " Black" ...
    ##  $ sex          : chr  " Male" " Male" " Male" " Male" ...
    ##  $ capitalgain  : int  2174 0 0 0 0 0 0 0 14084 5178 ...
    ##  $ capitalloss  : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ hoursperweek : int  40 13 40 40 40 40 16 45 50 40 ...
    ##  $ nativecountry: chr  " United-States" " United-States" " United-States" " United-States" ...
    ##  $ over50k      : chr  " <=50K" " <=50K" " <=50K" " <=50K" ...

``` r
barplot(table(census$over50k))
```

![](Predicting%20Earnings%20from%20Census%20Data_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
x = table(census$over50k)
x[1]/nrow(census)
```

    ##     <=50K 
    ## 0.7593658

We can see that 75% of the records have earned less than or equal to
50k.  
Let’s start by building a logistic regression model.

### Splitting the Data

``` r
library(caTools)
set.seed(123)
spl = sample.split(census$over50k, SplitRatio = 0.6)
train = subset(census, spl == TRUE)
test = subset(census, spl == FALSE)
```

### Building a Logistic Regression Model

``` r
train$over50k = as.factor(train$over50k)
test$over50k = as.factor(test$over50k)
logModel = glm(over50k ~ ., data = train, family = "binomial")
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
summary(logModel)
```

    ## 
    ## Call:
    ## glm(formula = over50k ~ ., family = "binomial", data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -5.0356  -0.4897  -0.1746  -0.0194   3.6231  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                            Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                              -6.784e+00  1.004e+00  -6.759 1.39e-11 ***
    ## age                                       2.231e-02  2.181e-03  10.227  < 2e-16 ***
    ## workclass Federal-gov                     1.216e+00  2.046e-01   5.941 2.83e-09 ***
    ## workclass Local-gov                       5.105e-01  1.879e-01   2.718 0.006576 ** 
    ## workclass Never-worked                   -9.354e+00  8.209e+02  -0.011 0.990909    
    ## workclass Private                         6.339e-01  1.685e-01   3.762 0.000169 ***
    ## workclass Self-emp-inc                    7.674e-01  2.003e-01   3.832 0.000127 ***
    ## workclass Self-emp-not-inc                7.604e-02  1.841e-01   0.413 0.679614    
    ## workclass State-gov                       2.524e-01  2.043e-01   1.236 0.216640    
    ## workclass Without-pay                    -1.292e+01  5.611e+02  -0.023 0.981630    
    ## education 11th                           -1.358e-01  2.755e-01  -0.493 0.622205    
    ## education 12th                            1.510e-01  3.730e-01   0.405 0.685497    
    ## education 1st-4th                        -6.092e-01  6.544e-01  -0.931 0.351887    
    ## education 5th-6th                        -4.937e-01  4.430e-01  -1.115 0.265002    
    ## education 7th-8th                        -6.369e-01  2.987e-01  -2.132 0.032999 *  
    ## education 9th                            -8.778e-01  3.828e-01  -2.293 0.021834 *  
    ## education Assoc-acdm                      1.345e+00  2.235e-01   6.021 1.74e-09 ***
    ## education Assoc-voc                       1.169e+00  2.141e-01   5.461 4.73e-08 ***
    ## education Bachelors                       1.827e+00  1.969e-01   9.278  < 2e-16 ***
    ## education Doctorate                       3.013e+00  2.794e-01  10.782  < 2e-16 ***
    ## education HS-grad                         7.082e-01  1.915e-01   3.699 0.000216 ***
    ## education Masters                         2.142e+00  2.112e-01  10.145  < 2e-16 ***
    ## education Preschool                      -2.170e+01  4.067e+02  -0.053 0.957456    
    ## education Prof-school                     2.518e+00  2.531e-01   9.948  < 2e-16 ***
    ## education Some-college                    1.038e+00  1.945e-01   5.334 9.59e-08 ***
    ## maritalstatus Married-AF-spouse           3.396e+00  6.693e-01   5.074 3.90e-07 ***
    ## maritalstatus Married-civ-spouse          2.260e+00  3.516e-01   6.429 1.28e-10 ***
    ## maritalstatus Married-spouse-absent       1.415e-01  2.920e-01   0.484 0.628101    
    ## maritalstatus Never-married              -5.889e-01  1.167e-01  -5.048 4.46e-07 ***
    ## maritalstatus Separated                  -9.378e-02  2.189e-01  -0.428 0.668344    
    ## maritalstatus Widowed                     2.421e-01  2.067e-01   1.171 0.241588    
    ## occupation Adm-clerical                   1.115e-01  1.306e-01   0.853 0.393401    
    ## occupation Armed-Forces                  -7.680e-01  1.721e+00  -0.446 0.655445    
    ## occupation Craft-repair                   2.224e-01  1.120e-01   1.987 0.046972 *  
    ## occupation Exec-managerial                8.771e-01  1.157e-01   7.581 3.43e-14 ***
    ## occupation Farming-fishing               -9.687e-01  1.931e-01  -5.018 5.23e-07 ***
    ## occupation Handlers-cleaners             -5.997e-01  1.888e-01  -3.176 0.001495 ** 
    ## occupation Machine-op-inspct             -8.503e-02  1.382e-01  -0.615 0.538414    
    ## occupation Other-service                 -9.622e-01  1.726e-01  -5.574 2.49e-08 ***
    ## occupation Priv-house-serv               -3.343e+00  2.266e+00  -1.475 0.140204    
    ## occupation Prof-specialty                 6.476e-01  1.241e-01   5.220 1.79e-07 ***
    ## occupation Protective-serv                5.578e-01  1.738e-01   3.211 0.001325 ** 
    ## occupation Sales                          3.796e-01  1.192e-01   3.185 0.001449 ** 
    ## occupation Tech-support                   6.927e-01  1.593e-01   4.349 1.37e-05 ***
    ## occupation Transport-moving                      NA         NA      NA       NA    
    ## relationship Not-in-family                6.227e-01  3.480e-01   1.790 0.073529 .  
    ## relationship Other-relative              -2.641e-01  3.188e-01  -0.828 0.407438    
    ## relationship Own-child                   -5.159e-01  3.462e-01  -1.490 0.136150    
    ## relationship Unmarried                    4.945e-01  3.691e-01   1.340 0.180330    
    ## relationship Wife                         1.439e+00  1.356e-01  10.614  < 2e-16 ***
    ## race Asian-Pac-Islander                   5.385e-01  3.645e-01   1.478 0.139538    
    ## race Black                                4.012e-01  3.082e-01   1.302 0.193046    
    ## race Other                                9.007e-02  4.824e-01   0.187 0.851877    
    ## race White                                5.228e-01  2.937e-01   1.780 0.075105 .  
    ## sex Male                                  8.812e-01  1.057e-01   8.336  < 2e-16 ***
    ## capitalgain                               3.207e-04  1.328e-05  24.150  < 2e-16 ***
    ## capitalloss                               7.059e-04  4.964e-05  14.220  < 2e-16 ***
    ## hoursperweek                              3.188e-02  2.143e-03  14.877  < 2e-16 ***
    ## nativecountry Canada                     -1.241e+00  8.935e-01  -1.389 0.164825    
    ## nativecountry China                      -2.053e+00  9.197e-01  -2.233 0.025569 *  
    ## nativecountry Columbia                   -4.309e+00  1.451e+00  -2.970 0.002980 ** 
    ## nativecountry Cuba                       -1.908e+00  9.500e-01  -2.009 0.044580 *  
    ## nativecountry Dominican-Republic         -2.846e+00  1.383e+00  -2.057 0.039674 *  
    ## nativecountry Ecuador                    -2.265e+00  1.159e+00  -1.954 0.050661 .  
    ## nativecountry El-Salvador                -3.072e+00  1.073e+00  -2.862 0.004208 ** 
    ## nativecountry England                    -1.841e+00  9.235e-01  -1.993 0.046249 *  
    ## nativecountry France                     -1.314e+00  1.046e+00  -1.256 0.209166    
    ## nativecountry Germany                    -1.400e+00  8.970e-01  -1.560 0.118653    
    ## nativecountry Greece                     -3.995e+00  1.393e+00  -2.867 0.004139 ** 
    ## nativecountry Guatemala                  -1.512e+00  1.166e+00  -1.298 0.194433    
    ## nativecountry Haiti                      -1.179e+00  1.233e+00  -0.956 0.338904    
    ## nativecountry Holand-Netherlands         -1.358e+01  1.455e+03  -0.009 0.992553    
    ## nativecountry Honduras                   -1.318e+01  4.959e+02  -0.027 0.978797    
    ## nativecountry Hong                       -1.511e+00  1.104e+00  -1.368 0.171233    
    ## nativecountry Hungary                    -2.055e+00  1.312e+00  -1.567 0.117132    
    ## nativecountry India                      -1.926e+00  8.898e-01  -2.165 0.030412 *  
    ## nativecountry Iran                       -1.416e+00  9.685e-01  -1.462 0.143674    
    ## nativecountry Ireland                    -9.111e-01  1.118e+00  -0.815 0.415019    
    ## nativecountry Italy                      -7.081e-01  9.520e-01  -0.744 0.456975    
    ## nativecountry Jamaica                    -3.041e+00  1.149e+00  -2.647 0.008127 ** 
    ## nativecountry Japan                      -9.604e-01  9.931e-01  -0.967 0.333506    
    ## nativecountry Laos                       -2.073e+00  1.196e+00  -1.733 0.083022 .  
    ## nativecountry Mexico                     -2.217e+00  8.784e-01  -2.524 0.011599 *  
    ## nativecountry Nicaragua                  -1.523e+01  2.749e+02  -0.055 0.955815    
    ## nativecountry Outlying-US(Guam-USVI-etc) -1.525e+01  4.369e+02  -0.035 0.972157    
    ## nativecountry Peru                       -2.655e+00  1.376e+00  -1.930 0.053668 .  
    ## nativecountry Philippines                -1.113e+00  8.500e-01  -1.310 0.190251    
    ## nativecountry Poland                     -1.737e+00  9.532e-01  -1.822 0.068421 .  
    ## nativecountry Portugal                   -1.479e+01  2.854e+02  -0.052 0.958665    
    ## nativecountry Puerto-Rico                -1.716e+00  9.780e-01  -1.754 0.079350 .  
    ## nativecountry Scotland                   -1.445e+01  7.116e+02  -0.020 0.983805    
    ## nativecountry South                      -3.333e+00  9.430e-01  -3.534 0.000409 ***
    ## nativecountry Taiwan                     -1.811e+00  1.040e+00  -1.741 0.081741 .  
    ## nativecountry Thailand                   -2.303e+00  1.199e+00  -1.922 0.054649 .  
    ## nativecountry Trinadad&Tobago            -2.397e+00  1.382e+00  -1.735 0.082787 .  
    ## nativecountry United-States              -1.699e+00  8.358e-01  -2.033 0.042010 *  
    ## nativecountry Vietnam                    -3.291e+00  1.134e+00  -2.903 0.003696 ** 
    ## nativecountry Yugoslavia                 -2.063e+00  1.206e+00  -1.711 0.087115 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 21175  on 19186  degrees of freedom
    ## Residual deviance: 11952  on 19090  degrees of freedom
    ## AIC: 12146
    ## 
    ## Number of Fisher Scoring iterations: 14

We can see that we have many significant variables to the model. Let’s
see the accuracy.

``` r
library(knitr)
predglm = predict(logModel, newdata = test, type = "response")
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == : prediction
    ## from a rank-deficient fit may be misleading

``` r
x = table(test$over50k, predglm >= 0.5)
kable(x)
```

|        | FALSE | TRUE |
|:-------|------:|-----:|
| \<=50K |  9041 |  672 |
| \>50K  |  1264 | 1814 |

``` r
(x[1]+x[4])/sum(x)
```

    ## [1] 0.8486436

The accuracy is almost 85%, not bad! Now, let’s compare it to a base
line model accuracy of the testing set.

``` r
kable(table(test$over50k))
```

| Var1   | Freq |
|:-------|-----:|
| \<=50K | 9713 |
| \>50K  | 3078 |

``` r
table(test$over50k)[1]/nrow(test)
```

    ##     <=50K 
    ## 0.7593621

We can see that a base line model on the testing set that always
predicts less than or equal to 50k is correct almost 76% of the times.
Meaning, the logistic regression model beats this by only 9%.  
Let’s see the AUC of the logistic model

``` r
#AUC calculations
library(ROCR)
ROCRpred = prediction(predglm, test$over50k)
as.numeric(performance(ROCRpred, "auc")@y.values)
```

    ## [1] 0.9000115

``` r
#ROC Plot
ROCplotlog = performance(ROCRpred, "tpr", "fpr")
plot(ROCplotlog)
```

![](Predicting%20Earnings%20from%20Census%20Data_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

Excellent! Let’s see if we can further improve the model.

### Building a CART Model

Now that we have a model that can serve as good base line for
understanding the siginificant variables, we can build a tree to see
what variables are more important than the others.

``` r
library(rpart)
library(rpart.plot)
CARTmodel = rpart(over50k ~ ., data = train, method = "class")
prp(CARTmodel)
```

![](Predicting%20Earnings%20from%20Census%20Data_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
Great! This is the power of a CART model. with a tree like this with
only 4 splits, we can intrepret that relationship is the most
significant variable followed by both capital gain and education
level.  
Let’s see how accurate the CART model is.

``` r
predCART = predict(CARTmodel, newdata = test, type = "class")
x = table(test$over50k, predCART)
kable(x)
```

|        | \<=50K | \>50K |
|:-------|-------:|------:|
| \<=50K |   9142 |   571 |
| \>50K  |   1477 |  1601 |

``` r
(x[1]+x[4]) / sum(x)
```

    ## [1] 0.8398874

We can see that the CART model has an accuracy of almost 84%. And while
it is lower than our logistic regression model, sometimes it is a good
idea to trade some of the accuracy for the interpretabilty of the CART
model.

### ROC and AUC of the CART model

``` r
#AUC 
predCART1 = predict(CARTmodel, newdata = test)
ROCRpredtree = prediction(predCART1[,2], test$over50k)
as.numeric(performance(ROCRpredtree, "auc")@y.values)
```

    ## [1] 0.8402365

``` r
#ROC Plot
rocplot = performance(ROCRpredtree, "tpr","fpr")
plot(rocplot)
```

![](Predicting%20Earnings%20from%20Census%20Data_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

***We can observe that the ROC curve of the tree is less smooth than the
logstic regression one, that is shown before. In fact, we can see that
there exsist 5 breaking points on the curve each corresponding to end
bucket / leaf of the tree. Furthermore, it is common for trees to have
breaking points because the probabilties of the CART model takes only a
handful of values.***

The AUC of the CART model is 0.84. Again, suggesting that the logistic
regression is performing better.

### Selecting CP by Cross Validation

We will use k-fold cross validation with 10 folds to determine the
optimal value of the cp (complexity parameter) for our CART model. This
involves dividing the data into 10 equal-sized folds and training the
model on 9 of the folds, using the remaining fold as the validation set.
The process is repeated 10 times, with each fold serving as the
validation set once. The cp value that results in the best performance
(as measured by some metric, such as accuracy or F1 score) is chosen as
the final value for the model.

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
set.seed(123)
fitControl = trainControl(method = "cv", number = 10)
cartGrid = expand.grid(.cp = seq(0.002, 0.1, 0.002))
train(over50k ~. , data = train, method = "rpart",trControl = fitControl, tuneGrid = cartGrid) 
```

    ## CART 
    ## 
    ## 19187 samples
    ##    12 predictor
    ##     2 classes: ' <=50K', ' >50K' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 17269, 17268, 17268, 17269, 17268, 17268, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp     Accuracy   Kappa    
    ##   0.002  0.8586017  0.5818970
    ##   0.004  0.8562564  0.5728049
    ##   0.006  0.8522432  0.5560285
    ##   0.008  0.8519303  0.5537328
    ##   0.010  0.8510966  0.5503582
    ##   0.012  0.8490118  0.5476791
    ##   0.014  0.8490118  0.5481041
    ##   0.016  0.8484383  0.5470157
    ##   0.018  0.8484383  0.5470157
    ##   0.020  0.8465618  0.5380425
    ##   0.022  0.8447901  0.5287526
    ##   0.024  0.8439040  0.5210003
    ##   0.026  0.8439040  0.5210003
    ##   0.028  0.8439040  0.5210003
    ##   0.030  0.8439040  0.5210003
    ##   0.032  0.8439040  0.5210003
    ##   0.034  0.8426013  0.5159514
    ##   0.036  0.8387967  0.5010671
    ##   0.038  0.8327512  0.4740825
    ##   0.040  0.8259751  0.4342498
    ##   0.042  0.8258187  0.4308641
    ##   0.044  0.8258187  0.4308641
    ##   0.046  0.8258187  0.4308641
    ##   0.048  0.8258187  0.4308641
    ##   0.050  0.8258187  0.4308641
    ##   0.052  0.8215457  0.4012943
    ##   0.054  0.8181585  0.3751639
    ##   0.056  0.8168550  0.3627600
    ##   0.058  0.8149265  0.3404014
    ##   0.060  0.8132066  0.3146268
    ##   0.062  0.8132066  0.3146268
    ##   0.064  0.8132066  0.3146268
    ##   0.066  0.8132066  0.3146268
    ##   0.068  0.8132066  0.3146268
    ##   0.070  0.8074734  0.2821501
    ##   0.072  0.7962157  0.2171280
    ##   0.074  0.7962157  0.2171280
    ##   0.076  0.7841782  0.1466111
    ##   0.078  0.7593684  0.0000000
    ##   0.080  0.7593684  0.0000000
    ##   0.082  0.7593684  0.0000000
    ##   0.084  0.7593684  0.0000000
    ##   0.086  0.7593684  0.0000000
    ##   0.088  0.7593684  0.0000000
    ##   0.090  0.7593684  0.0000000
    ##   0.092  0.7593684  0.0000000
    ##   0.094  0.7593684  0.0000000
    ##   0.096  0.7593684  0.0000000
    ##   0.098  0.7593684  0.0000000
    ##   0.100  0.7593684  0.0000000
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.002.

We can see the optimal cp is 0.002. Now let’s build a new CART model
with this value.

``` r
CARTmodelcv = rpart(over50k ~ ., data = train, method = "class", cp = 0.002)
prp(CARTmodelcv)
```

![](Predicting%20Earnings%20from%20Census%20Data_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

Let’s see the accuracy of it.

``` r
predCARTcv = predict(CARTmodelcv, newdata = test, type = "class")
x = table(test$over50k, predCARTcv)
kable(x)
```

|        | \<=50K | \>50K |
|:-------|-------:|------:|
| \<=50K |   9138 |   575 |
| \>50K  |   1286 |  1792 |

``` r
#Accuracy
(x[1]+x[4])/sum(x)
```

    ## [1] 0.8545071

### Building a Random Forest model.

``` r
library(randomForest)
```

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
set.seed(123)
RFmodel = randomForest(over50k~ ., data= train)
```

``` r
predRFm = predict(RFmodel, newdata = test)
x = table(test$over50k, predRFm)
kable(x)
```

|        | \<=50K | \>50K |
|:-------|-------:|------:|
| \<=50K |   9167 |   546 |
| \>50K  |   1223 |  1855 |

``` r
(x[1]+x[4])/sum(x)
```

    ## [1] 0.8616996

The accuracy of the random forest model is 86% !.

We can assess the importance of different variables in a random forest
model by examining how frequently they are selected for splits when the
model is trained. This can be done by aggregating the number of times
each variable is chosen for a split across all of the trees in the
model. This gives us a measure of the overall importance of each
variable in the model.

``` r
vu = varUsed(RFmodel, count = TRUE)
vusorted = sort(vu, decreasing = FALSE, index.return = TRUE)
dotchart(vusorted$x, names(RFmodel$forest$xlevels[vusorted$ix]))
```

![](Predicting%20Earnings%20from%20Census%20Data_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

The varUsed() function is used to calculate the variable importance for
each variable in the model. The “count = TRUE” argument specifies that
the importance should be calculated based on the number of times each
variable is used in the model (as opposed to the reduction in impurity).

The resulting variable importance values are then sorted in ascending
order using the sort() function, with the “decreasing = FALSE” argument
specifying that the values should be sorted in ascending order. The
“index.return = TRUE” argument specifies that the function should return
the indices of the sorted values as well as the sorted values
themselves.

Finally, the dotchart() function is used to create a dot chart
visualization of the sorted variable importance values. The first
argument to the function is the sorted variable importance values
(vusorted\$x), and the second argument is a list of the variable names
(names(RFmodel\$forest\$xlevels\[vusorted\$ix\])). This creates a dot
chart with the variable names on the x-axis and the importance values on
the y-axis, with the variables ordered from least to most important.

***We can see the most important variable in terms of the number of
splits in the random forest model is Age***

Another way to measure the importance of variables in a random forest
model is to calculate the average reduction in impurity that results
from using each variable for a split. Impurity is a measure of how mixed
or homogeneous the data in a particular node or leaf of the tree is.
When a split is performed using a particular variable, the impurity is
typically reduced. By averaging the reduction in impurity that results
from using a variable for a split across all of the trees in the forest
and all of the times the variable is selected for a split, we can get a
measure of the overall importance of that variable in the model.

``` r
varImpPlot(RFmodel)
```

![](Predicting%20Earnings%20from%20Census%20Data_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

***For instance, it can be see that the most important variable in terms
of reduction of impurity is Capitalgain.***
