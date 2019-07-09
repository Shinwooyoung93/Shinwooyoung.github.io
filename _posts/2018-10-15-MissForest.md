---
layout: post
title:  "Statistical method project - MissForest"
date:   2018-10-15
use_math: true
tags:
 - R
 - korean
 - Report
---

# MissForest

## 1. Intro

결측값을 추정하는 일은 데이터 분석에서 매우 중요한 역할을 한다. 가장 단순한 mean imputation부터 많이 쓰이는 MICE (Van Buuren and Oudshoorn, 1999)까지 다양한 추정방법이 존재한다. 그러나 categorical data와 continous data가 혼합된 mixed data의 경우, 결측값을 추정하기란 쉬운일이 아니다. 


Mixed data의 경우 흔히들 MICE를 사용하는데 이 경우 full multivariate distribution이 존재해야하며, 결측값들은 full distribution을 기반으로하는 conditional distributions에서 추출되어야한다. 그 정도로 복잡한 가정을 기반으로 하는 반면에, 비모수적인 방법이며 random forest를 기반으로 하는 MissForest는 많은 이점을 지닌다. 또한 continous data와 categorical data와 같이 데이터의 유형에서 자유로은 random forest를 기반으로 하므로 MissForest의 결측값 추정방법은 매우 자연스럽다. 


2절에서는 MissForest알고리즘에 대해 설명하고 3절에서는 MissForest의 알고리즘을 설명하기 위해 간단한 R예제를 보이겠으며 4절에서는 MissForest와 MICE의 성능 비교를, 5절에서는 KNN imputation과의 성능 비교를, 6절에서는 EM imputation과 성능 비교를 하면서 마치도록 하겠다.

## 2. Algorithm for MissForest

### 1) Organize terms
- $\textbf{i}_{mis}^{(s)} \subseteq \{1, \cdots, n\}$
- The observed values of variable $X_s,$ denoted by $y_{obs}^{(s)}$
- The missing values of variable $X_s,$ denoted by $y_{mis}^{(s)}$
- The variables other than $X_s$ with observations $\textbf{i}_{obs}^{(s)} = \{1, \cdots, n\}$ \ $\textbf{i}_{mis}^{(s)}$ denoted by $x_{obs}^{(s)}$
- The variables other than $X_s$ with observations $\textbf{i}_{mis}^{(s)}$ denoted by $x_{mis}^{(s)}$
- Note that $x_{obs}^{(s)}$ is typically not completely observed since the index $\textbf{i}_{obs}^{(s)}$ corresponds to the observed values of the variable $X_s$
- Likewise, $x_{mis}^{(s)}$ is typically not completely missing

### 2) Algorithm

**Algorithm** : Imputate missing values with **random forest**

**Require** : $X$ an $n \times p$, stopping criterion $\gamma$

1. Make initial guess for missing values(usually use mean imputation);
2. $\textbf{k} \leftarrow \:$ vector of sorted indices of clumns in $X, \:$w.r.t. in creasing amount of missing values;
3. **while** not $\gamma$ do
4. $\quad X_{old}^{imp} \leftarrow$ store previously imputed matrix;
5. $\quad$**for** $s$ in $\textbf{k}$ **do**
6. $\quad \quad$Fit a random forest : $y_{obs}^{(s)} ~ x_{obs}^{(s)}$;
7. $\quad \quad$Predict $y_{mis}^{(s)}$ using $x_{mis}^{(s)}$;
8. $\quad \quad X_{new}^{imp} \leftarrow$ update imputed matrix, using predicted $y_{mis}^{(s)}$;
9. $\quad$ **end for** 
10. $\quad$update $\gamma$
11. **end while**
12. **return** the imputed matrix $X^{imp}$ 

### 3) Criterion $\gamma$
The stopping criterion $\gamma$ is difference for the set of continuous variables.

\begin{align*}
& \nabla_N = \frac{\sum_{j \in N}(X_{new}^{imp} - X_{old}^{imp})^2}{\sum_{j \in N}(X_{new}^{imp})^2}, \enspace N \: \text{is continuous variables}\\
& \nabla_F = \frac{\sum_{j \in F}\sum_{i = 1}^{n}X_{new}^{imp} \neq X_{old}^{imp}}{\# NA}, \enspace F \: \text{is categorical variables}
\end{align*}

In both cases that $\nabla_N$ and $\nabla_F$, good performance leads to a value close to 0, bad performance to a value around 1

## 3. R-code for simple example

ISLR library에 있는 Carseats데이터이다. 간단하게 데이터의 형태를 보면

```r
rm(list = ls())
set.seed(1)
library(ISLR)
library(ranger)
library(randomForest)
data <- Carseats
colnames(data)<-c("Sales","CompP","In","Ad","Pop","Price","Shelve","Age","Ed","Ur","US")
head(data);data <- Carseats
```
```r
## Sales CompP In Ad Pop Price Shelve Age Ed Ur US
## 1 9.50 138 73 11 276 120 Bad 42 17 Yes Yes
## 2 11.22 111 48 16 260 83 Good 65 10 Yes Yes
## 3 10.06 113 35 10 269 80 Medium 59 12 Yes Yes
## 4 7.40 117 100 4 466 97 Medium 55 14 Yes Yes
## 5 4.15 141 64 3 340 128 Bad 38 13 Yes No
## 6 10.81 124 113 13 501 72 Bad 78 16 No Yes
```

Sales, CompPrice, Income, Advertising, Population, Price는 continuous variables이고 그 외의 변수는 categorical variables이다. 
Age의 변수를 20대, 30대, 40대, 50대, 60대 이상으로 총 5개의 범주로 바꾸어 분석할 것이다. 데이터를 전부 수치화 시켜준다.

```r
# transform data type
# ShelveLoc
ShelveLoc.level <- c("Bad", "Medium", "Good")
for(i in 1:3){
  data$ShelveLoc <- ifelse(data$ShelveLoc == ShelveLoc.level[i], i, data$ShelveLoc)}

# Age
Age.level <- c(30, 40, 50, 60)
data$Age <- ifelse(data$Age < 30, 1, data$Age)
for(i in 2:4){
  data$Age <- ifelse(Age.level[i - 1] <= data$Age & data$Age < Age.level[i], i, data$Age)}
data$Age <- ifelse(data$Age >= 60, 5, data$Age)

# Education
Education.level <- 10:18
for(i in 1:9){
  data$Education <- ifelse(data$Education == Education.level[i], i, data$Education)}

# Urban
data$Urban <- factor(as.numeric(data$Urban)) # No is 1

# Us
data$US <- factor(as.numeric(data$US)) # No is 1

for(j in 7:11){data[,j] <- factor(data[,j])}
colnames(data)<-c("Sales","CompP","In","Ad","Pop","Price","Shelve","Age","Ed","Ur","US")
head(data)
```
```r
## Sales CompP In Ad Pop Price Shelve Age Ed Ur US
## 1 9.50 138 73 11 276 120 1 3 8 2 2
## 2 11.22 111 48 16 260 83 2 5 1 2 2
## 3 10.06 113 35 10 269 80 3 4 3 2 2
## 4 7.40 117 100 4 466 97 3 4 5 2 2
## 5 4.15 141 64 3 340 128 1 2 4 2 1
## 6 10.81 124 113 13 501 72 1 5 7 1 2
```

간단한 예제이므로 (1, 1)성분과 (2, 4)성분을 NA로 바꾸어 추정과정을 설명하겠다. 이는 성분을 알려주는 코드이다.

```r
raw.data1 <- data[1,1];raw.data2 <- data[2, 4]
data[1, 1] <- NA;data[2, 4] <- NA

na.num <- which(is.na(data) == 1)
na.index <- matrix(NA, length(na.num), 2)
na.index[,1] <- na.num%%400 # means row
na.index[,2] <- ceiling(na.num/400) # means col
na.index
```
```r
## [,1] [,2]
## [1,] 1 1
## [2,] 2 4
```

시작을 위해 criterion $\gamma$ 함수를 정의해주고 mean imputation으로 먼저 결측치를 채워준다. 
또한 criterion $\gamma$ 의 하한값 $\epsilon$ 을 다음과 같이 설정한다.

```r
# initializing
gamma.func1 <- function(x.new, x.old){sum((x.new - x.old)^2)/sum(x.new^2)}
gamma.func2 <- function(x.new, x.old){sum(x.new != x.old)/length(na.num)}
max.iter <- 20
X.new <- data

for(i in 1:nrow(na.index)){
  sel.num <- na.index[i,]
  X.new[sel.num[1],sel.num[2]] <- mean(na.omit(X.new[,sel.num[2]]))
}
epsilon <- 1e-5

# start simulation
for(iter in 1:max.iter){

  X.old <- X.new
  rate <- rep(NA, ncol(X.old))
    
  for(s in 1:nrow(na.index)){
    temp.train.y <- X.old[-c(na.index[s,1]), na.index[s,2]]
    temp.train.X <- X.old[-c(na.index[s,1]), -na.index[s,2]]
    temp.test.y <- X.old[c(na.index[s,1]), na.index[s,2]]
    temp.test.X <- X.old[c(na.index[s,1]), -na.index[s,2]]
    #rf.fit <- randomForest(temp.train.y~., data = temp.train.X, ntree = 100)
    #pred <- predict(rf.fit, temp.test.X)
    rf.fit <- ranger(temp.train.y ~., data = temp.train.X, num.tree = 100)
    pred <- predict(rf.fit, temp.test.X)$predictions
    sel.num <- na.index[s,]
    X.new[sel.num[1], sel.num[2]] <- pred
  }
  
  for(j in 1:ncol(X.new)){
    if(is.factor(X.new[,j]) == T){
      rate[j] <- gamma.func2(X.new[,j], X.old[,j])
    }else rate[j] <- gamma.func1(X.new[,j], X.old[,j])
  }
  cat(paste(iter, "iter", "&", "rate = ", sum(rate)), "\n")
  if(sum(rate) <= epsilon) break
}
```
```r
## 1 iter & rate = 0.000417302473814881
## 2 iter & rate = 1.34861440614928e-05
## 3 iter & rate = 3.24299861955174e-05
## 4 iter & rate = 3.14072141016627e-05
## 5 iter & rate = 1.04844733001558e-06
```
```r
X.new[1:4,]
```
```r
## Sales CompP In Ad Pop Price Shelve Age Ed Ur US
## 1 6.53261 138 73 11.000000 276 120 1 3 8 2 2
## 2 11.22000 111 48 9.662833 260 83 2 5 1 2 2
## 3 10.06000 113 35 10.000000 269 80 3 4 3 2 2
## 4 7.40000 117 100 4.000000 466 97 3 4 5 2 2
```

MissForest library는 \textbf{random forest}로 짜여져있지만 간단한 예제 데이터이므로 속도와 예측력을 높이기 위해 \textbf{ranger} library를 사용하였다. 

```r
output <- rbind(c(raw.data1, raw.data2),
                c(X.new[1, 1], X.new[2, 4]),
                c(mean(na.omit(data[,1])),
                  mean(na.omit(data[,4]))))
rownames(output) <- c("raw", "imp", "mean")
colnames(output) <- c("(1, 1)", "(2, 4)")
output
```
```r
## (1, 1) (2, 4)
## raw 9.500000 16.000000
## imp 6.532610 9.662833
## mean 7.491303 6.611529
```

## 4. R-code for MissForest vs MICE

위의 데이터에서 조금 더 복잡하게 결측치를 만들어서 MICE와 비교해보도록 하겠다.

```r
rm(list = ls())
set.seed(1)
library(ISLR)
library(missForest)
library(mice)
data <- Carseats;#attach(data)
# transform data type
# ShelveLoc
ShelveLoc.level <- c("Bad", "Medium", "Good")
for(i in 1:3){
  data$ShelveLoc <- ifelse(data$ShelveLoc == ShelveLoc.level[i], i, data$ShelveLoc)}

# Age
Age.level <- c(30, 40, 50, 60)
data$Age <- ifelse(data$Age < 30, 1, data$Age)
for(i in 2:4){
  data$Age <- ifelse(Age.level[i - 1] <= data$Age & data$Age < Age.level[i], i, data$Age)}
data$Age <- ifelse(data$Age >= 60, 5, data$Age)

# Education
Education.level <- 10:18
for(i in 1:9){
  data$Education <- ifelse(data$Education == Education.level[i], i, data$Education)}

# Urban
data$Urban <- factor(as.numeric(data$Urban)) # No is 1

# Us
data$US <- factor(as.numeric(data$US)) # No is 1

for(j in 7:11){data[,j] <- factor(data[,j])}
colnames(data)<-c("Sales","CompP","In","Ad","Pop","Price","Shelve","Age","Ed","Ur","US")
X.true <- data
head(data)
```
```r
## Sales CompP In Ad Pop Price Shelve Age Ed Ur US
## 1 9.50 138 73 11 276 120 1 3 8 2 2
## 2 11.22 111 48 16 260 83 2 5 1 2 2
## 3 10.06 113 35 10 269 80 3 4 3 2 2
## 4 7.40 117 100 4 466 97 3 4 5 2 2
## 5 4.15 141 64 3 340 128 1 2 4 2 1
## 6 10.81 124 113 13 501 72 1 5 7 1 2
```

```r
# set NA for imputation
random.na <- sample(1:(nrow(data)*ncol(data)), 200, replace = F)
na.index <- matrix(NA, length(random.na), 2)
na.index[,1] <- random.na%%400 # means row
na.index[,2] <- ceiling(random.na/400) # means col
for(i in 1:nrow(na.index)){sel.num <- na.index[i,];data[sel.num[1], sel.num[2]] <- NA;}
```

주의해야할 점은 위의 결측치 데이터는 mixed data이기 때문에 어떠한 결측치의 distribution을 가정할 수 없는 상황이다. 여기에서 MICE의 가정인 distribution으로 인한 신뢰성 문제가 드러난다. 그러나 이러한 분포가정이 필요없는 MissForest는 그에 반해 신뢰성을 가질 수 있게 된다. 이에 대한 것을 무시하고 분석을 진행해보겠다.

```r
# fit
toc.mf <- Sys.time()
fit.mf <- missForest(data)
tic.mf <- Sys.time();time.mf <- tic.mf - toc.mf
toc.mc <- Sys.time()
fit.mc <- mice(data)
tic.mc <- Sys.time();time.mc <- tic.mc - toc.mc
# validation
gamma.func1 <- function(x.new, x.old){sum((x.new - x.old)^2)/sum(x.new^2)}
gamma.func2 <- function(x.new, x.old){sum(x.new != x.old)/nrow(na.index)}

X.imp1 <- fit.mf$ximp
X.imp2 <- complete(fit.mc)

delta.N1 <- gamma.func1(X.imp1[,1:6], X.true[,1:6])
delta.N2 <- gamma.func1(X.imp2[,1:6], X.true[,1:6])
delta.F1 <- gamma.func2(X.imp1[,-(1:6)], X.true[,-(1:6)])
delta.F2 <- gamma.func2(X.imp2[,-(1:6)], X.true[,-(1:6)])

mf <- c(time.mf, delta.N1, delta.F1)
mc <- c(time.mc, delta.N2, delta.F2)
output <- rbind(MissForest = mf, MICE = mc)
colnames(output) <- c("time", "continuous", "categorical")
output
```
```r
## time continuous categorical
## MissForest 5.974023 0.009791933 0.240
## MICE 12.933395 0.013018709 0.265
```

결측값을 200개로 늘린 결과가 위에 있으며 아래의 criterion $\gamma$ 들을 사용하였다.

속도나 continuous나 categorical에 대해서 전부 MissForest방법이 MICE보다 좋은 것을 확인하였다.

\begin{align*}
& \nabla_N = \frac{\sum_{j \in N}(X_{new}^{imp} - X_{old}^{imp})^2}{\sum_{j \in N}(X_{new}^{imp})^2}, \enspace N \: \text{is continuous variables}\\
& \nabla_F = \frac{\sum_{j \in F}\sum_{i = 1}^{n}X_{new}^{imp} \neq X_{old}^{imp}}{\# NA}, \enspace F \: \text{is categorical variables}
\end{align*}

In both cases that $\nabla_N$ and $\nabla_F$, good performance leads to a value close to 0, bad performance to a value around 1

또한 주목해야할 것은 결측값의 수에 주목해야하는데, 결측값의 수가 작을 경우(n = 100)와 결측값의 수가 클 경우(n = 200)을 비교한 표를 보면 확연한 차이가 있다는 것을 확인할 수 있다. 결측값의 수가 커질 수록 $\nabla_N$ 의 차이가 점점 커지는 것을 확인할 수 있다.

|  | |System.time | |$\nabla_N$ | |$\nabla_F$ | 
|:---|-|:---:|-|:---:|-|:---:|
|**MissForest(n = 300)**| |5.776722 | |0.01174297 | |0.2166667 |
|**MICE(n = 300)**| |16.054633 | |0.02131126 | |0.2166667 |
|**MissForest(n = 100)**| |8.379014 | |0.01384967 | |0.68 | 
|**MICE(n = 100)**| |20.563070 | |0.01762438 | |0.68 | 

## 5. R-code for MissForest vs KNN imputation

연속형 자료에 대해 imputation을 해야할 경우가 있다. 연속형 자료에 대해서 유클리디안 거리를 사용해 관측값들의 거리로 군집을 만들고, 이를 이용해 결측치를 추정하는 방식은 매우 자연스럽다. 따라서 군집화의 대표적인 방법인 KNN을 이용해 결측값을 채우는 라이브러리 knnImputation을 사용해 연속형 자료에 대해 결측치를 채워 비교해본다. 이때 주의해야할 점은 독립변수와 종속변수의 관계를 설정해야하지 않아야 한다는 것인데, 이는 군집의 특성에 기반하기 때문이다.

```r
rm(list = ls())
set.seed(1)
library(ISLR)
library(missForest)
library(DMwR)
library(MissMech)
data <- Carseats[,1:6] # except factor variables
X.true <- data
# set NA for imputation
random.na <- sample(1:(nrow(data)*ncol(data)), 50, replace = F)
na.index <- matrix(NA, length(random.na), 2)
na.index[,1] <- random.na%%400 # means row
na.index[,2] <- ceiling(random.na/400) # means col
for(i in 1:nrow(na.index)){sel.num <- na.index[i,];data[sel.num[1], sel.num[2]] <- NA;}
TestMCARNormality(data)
```
```r
## Call:
## TestMCARNormality(data = data)
##
## Number of Patterns: 4
##
## Total number of cases used in the analysis: 382
##
7
## Pattern(s) used:
## Sales CompPrice Income Advertising Population Price
## group.1 1 1 1 1 1 1
## group.2 1 1 NA 1 1 1
## group.3 1 1 1 1 NA 1
## group.4 1 1 1 NA 1 1
## Number of cases
## group.1 352
## group.2 10
## group.3 11
## group.4 9
##
##
## Test of normality and Homoscedasticity:
## -------------------------------------------
##
## Hawkins Test:
##
## P-value for the Hawkins test of normality and homoscedasticity: 0.2820002
##
## There is not sufficient evidence to reject normality
## or MCAR at 0.05 significance level
```

위의 결과는 결측치들이 MCAR가정을 만족한다는 것을 말하며 분석을 실시한다.

```r
# fit
toc.mf <- Sys.time()
fit.mf <- missForest(data)
tic.mf <- Sys.time();time.mf <- tic.mf - toc.mf
toc.ki <- Sys.time()
fit.ki <- knnImputation(data, k = 10)
tic.ki <- Sys.time();time.ki <- tic.ki - toc.ki

gamma.func1 <- function(x.new, x.old){sum((x.new - x.old)^2)/sum(x.new^2)}

X.imp1 <- fit.mf$ximp
X.imp2 <- fit.ki

delta.N1 <- gamma.func1(X.imp1, X.true)
delta.N2 <- gamma.func1(X.imp2, X.true)

mf <- c(time.mf, delta.N1)
ki <- c(time.ki, delta.N2)
output <- rbind(MissForest = mf, KNN = ki)
colnames(output) <- c("time", "continuous")
output
```
```r
## time continuous
## MissForest 1.7911808 0.003293605
## KNN 0.0219419 0.003942237
```

간단하게 결측치의 수가 적을 때(n = 50)에 대해선 MissForest방법이 KNN imputation방법보다 좋다는 것을 알 수 있었다. 
그러나 KNN기법에 기인한 추정방법은 다음과 같을 때 단점을 지니고 MissForest방법이 장점을 지닌다.

```r
rm(list = ls())
set.seed(1)
library(ISLR)
library(missForest)
library(DMwR)
library(MissMech)
data <- Carseats[,1:6] # except factor variables
X.true <- data

for(i in 1:nrow(data)){random.na <- sample(1:6, 1, replace = F)
  data[i,random.na] <- NA}
TestMCARNormality(data)
```
```r
## Warning: There is not sufficient number of complete cases.
## Dist.Free imputation requires a least 10 complete cases
## or 2*number of variables, whichever is bigger.
## imputation.method = normal will be used instead.
## Call:
## TestMCARNormality(data = data)
##
## Number of Patterns: 6
##
## Total number of cases used in the analysis: 400
##
## Pattern(s) used:
## Sales CompPrice Income Advertising Population Price
## group.1 1 NA 1 1 1 1
## group.2 1 1 NA 1 1 1
## group.3 1 1 1 NA 1 1
## group.4 1 1 1 1 1 NA
## group.5 NA 1 1 1 1 1
## group.6 1 1 1 1 NA 1
## Number of cases
## group.1 76
## group.2 83
## group.3 60
## group.4 64
## group.5 58
## group.6 59
##
##
## Test of normality and Homoscedasticity:
## -------------------------------------------
##
## Hawkins Test:
##
## P-value for the Hawkins test of normality and homoscedasticity: 0.4593001
##
## There is not sufficient evidence to reject normality
## or MCAR at 0.05 significance level
```

위의 결측치가 생긴 형태는 전체 관측값들이 적어도 1개의 결측치를 갖고 있을 경우이다. 
총 결측치의 갯수는 400개 이며, 결측값이 없는 관측치는 존재하지 않는다. 이곳에서 MissForest의 장점이 나타나는데, MissForest는 먼저 mean imputation방법으로 결측치를 채우고 train set을 형성해 추정하는 방식이었다면 KNN imputation은 결측값이 없는 관측값들을 train set으로 형성해 추정한다. 따라서 결측값의 수가 적더라도 관측값이 적어도 1개씩 결측값을 갖고 있다면 이 방법으로 추정하는 것은 불가능하다. 그에 따른 결측치의 수가 많을 때와 적어도 1개의 결측치를 갖는 dataset일 때(n = 100, 200, 400(*))에 대한 결과는 아래의 표와 같다.

#### 400(*) : 행이 400이고 랜덤하게 변수 중 1개가 결측인 경우

|  | |System.time | |$\nabla_N$ |
|:---|-|:---:|-|:---:|
|**MissForest(n = 400)**|| 3.38719583 || 0.03016432 |
|**KNN(n = 400)**|| invalid || invalid |
|**MissForest(n = 200)**|| 2.8766959 || 0.01578757 |
|**KNN(n = 200)**|| 0.2097521 || 0.01466621 |
|**MissForest(n = 100)**|| 2.41454792 || 0.007737714 |
|**KNN(n = 100)**|| 0.04053402 || 0.007329508 |
|**MissForest(n = 50)**|| 2.63248515 || 0.003250876 |
|**KNN(n = 50)**|| 0.04986715 || 0.003942237 |

위의 결과를 살펴보면 결측치의 갯수가 작을 때(n = 50)는 MissForest의 방식 성능이 좋다는 것을 알 수 있다. 
그러나 결측치의 수가 많아질수록 KNN imputation 성능이 좋다는 것을 알 수 있다. 
끝으로 현재 데이터는 dimension이 6에 불과하지만 관측값이 작은 데, high dimension에서 적어도 1개의 관측값이 결측치를 포함한다면 KNN imputation을 하는 것은 불가능한 반면 MissForest는 장점을 지닌다고 할 수 있다.

## 6. R-code for MissForest vs EM imputation

마지막으로 EM imputation으로 결측치 추정결과를 확인 해보겠다. 
EM algorithm을 활용해 imputation하는 방법은 응답된 자와 현재 반복에서의 모수 추정값을 이용하여 complete expectation log likelihood를 구한다. 그 다음 결측치에 대해 M-step과 E-step을 반복하는 알고리즘 방식이다. 

```r
rm(list = ls())
set.seed(1)
library(ISLR)
library(missForest)
library(Amelia)
library(MissMech)
data <- Carseats[,1:6] # except factor variables
X.true <- data

# set NA for imputation
random.na <- sample(1:(nrow(data)*ncol(data)), 50, replace = F)
na.index <- matrix(NA, length(random.na), 2)
na.index[,1] <- random.na%%400 # means row
na.index[,2] <- ceiling(random.na/400) # means col
for(i in 1:nrow(na.index)){sel.num <- na.index[i,];data[sel.num[1], sel.num[2]] <- NA;}
TestMCARNormality(data)
```
```r
## Call:
## TestMCARNormality(data = data)
##
## Number of Patterns: 4
##
## Total number of cases used in the analysis: 382
##
## Pattern(s) used:
## Sales CompPrice Income Advertising Population Price
## group.1 1 1 1 1 1 1
## group.2 1 1 NA 1 1 1
## group.3 1 1 1 1 NA 1
## group.4 1 1 1 NA 1 1
## Number of cases
## group.1 352
## group.2 10
## group.3 11
## group.4 9
##
##
## Test of normality and Homoscedasticity:
## -------------------------------------------
##
## Hawkins Test:
##
## P-value for the Hawkins test of normality and homoscedasticity: 0.2820002
##
## There is not sufficient evidence to reject normality
## or MCAR at 0.05 significance level
```

```r
# fit
toc.mf <- Sys.time()
fit.mf <- missForest(data)
tic.mf <- Sys.time();time.mf <- tic.mf - toc.mf
toc.em <- Sys.time()
fit.em <- amelia(data)
tic.em <- Sys.time();time.em <- tic.em - toc.em

gamma.func1 <- function(x.new, x.old){sum((x.new - x.old)^2)/sum(x.new^2)}

X.imp1 <- fit.mf$ximp
X.imp2 <- as.data.frame(fit.em$imputations[5])

delta.N1 <- gamma.func1(X.imp1, X.true)
delta.N2 <- gamma.func1(X.imp2, X.true)

mf <- c(time.mf, delta.N1)
em <- c(time.em, delta.N2)
output <- rbind(MissForest = mf, EM = em)
colnames(output) <- c("time", "continuous")
output
```
```r
## time continuous
## MissForest 1.81414986 0.003293605
## EM 0.04883695 0.007816180
```

EM으로 적합한 결과의 $\gamma$ 값은 매우 높다. 이는 imputation을 하는 데 있어 likelihood에 기반하므로 음의 값을 가질 수 있다는 점에서 차이가 나기 때문이다.

## Conclusion

우리는 지금까지의 진행과정에서 결측치 추정에 있어 MissForest보다 좋은 추정을 하는 경우는 결측치의 갯수가 작은 경우(n = 50) MICE 외에 존재하지 않았다. MICE의 경우 full multivariate distribution이 존재해야하며, 결측값들은 full distribution을 기반으로하는 conditional distributions에서 추출되어야한다. 그러므로 MICE의 성능이 좋다는 것은 실제로 판단할 수 없으며 여기에서 MissForest는 이점을 가진다. 어떠한 결측치의 형태에 있어 가정을 필요로 하지 않으며 가정을 만족한들, 다른 방식으로 하여금 성능을 보장받을 수 없기 때문이다. 마지막으로 정리하자면 MissForest는 비모수적인 방법으로 MICE와 EM알고리즘에 기반한 결측치 추정방식보다 가정에 대해 자유롭고 성능을 보장받는다. 또한 KNN imputation에 비해 결측값이 적을 때 성능을 보장받지만 결측치의 수가 많아질 수록 성능을 보장받지 못한다. 그러나 dimension이 큰 상황에서 결측치가 들어가 있지 않은 dataset에서 KNN imputation을 쓸 수 없다는 점에서 MissForest방식은 장점을 갖는다고 말할 수 있다.

## Reference

Marvin N.Wright and Andreas Ziegler (2017) **ranger** *A Fast Implementation of Random Forests for High Dimensional Data in C++ and R* . Universitat zu Lubeck

Van Buuren,S. and Oudshoorn,K. (1999) *Flexible Multivariate Inputation by* **MICE**. TNO Prevention Center, Leiden, The Netherlands

Daniel J.Stekhoven and Peter Buhlmann (2011) **MissForest** : *non-parametric missing value imputation for mixed-type data*. Department of Mathematics, ETH Zurich

N. S. Altman (1992) *An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression*.  Cornell University, Biometrics Unit

A.P.Dempster, N.M.Laird and D.B.Rubin (1977) *Maximum Likelihood from Incomplete Data via the* **EM** *Algorithm*. Havard University and Educational Testing Service

Daniel J. Stekhoven (2013). **missForest**: *Nonparametric Missing Value Imputation using Random Forest*. R package version 1.4.

Mortaza Jamshidian, Siavash Jalal, Camden Jansen (2014). **MissMech**: *An R Package for Testing Homoscedasticity, Multivariate Normality, and Missing Completely at Random (MCAR)*. Journal of Statistical Software, 56(6), 1-31. URL http://www.jstatsoft.org/v56/i06/.

Torgo, L. (2010). Data Mining with R, learning with case studies Chapman and Hall/CRC. URL: http://www.dcc.fc.up.pt/~ltorgo/DataMiningWithR

James Honaker, Gary King, Matthew Blackwell (2011). **Amelia II**: *A Program for Missing Data*. Journal of Statistical Software, 45(7), 1-47. URL http://www.jstatsoft.org/v45/i07/.

Gareth James, Daniela Witten, Trevor Hastie and Rob Tibshirani (2017). **ISLR**: *Data for an Introduction to Statistical Learning with Applications in R*. R package version 1.2. https://CRAN.R-project.org/package=ISLR
