---
layout: post
title:  "SHB contest"
date:   2018-11-13
use_math: true
tags:
 - R
 - korean
 - Contest
---

# SHB

## 1. Make test data

빅콘테스트의 금융분야에서 신한은행의 금융데이터를 활용한 "나의 금융생활 정보지수"개발 공모전에 참여하였다. 
고객이 본인의 정보를 입력하면 본인과 비슷한 사람들의 평균 금융거래정보를 보여주고 본인의 금융거래 정보를 입력하면 비슷한 사람들 중에 본인의 위치를 비교해 보여주는 시스템을 만드는 것이다. 
먼저 신한은행 측에서 요구한 과제를 살펴보면 

![](/assets/SHB/1.PNG)![](/assets/SHB/2.PNG)
![](/assets/SHB/3.PNG)![](/assets/SHB/4.PNG)

주어진 1만 7천여개의 데이터는 성별, 연령과 같은 '고객기본정보' 8개와 금융자산, 부동산자산 등과 같은 '금융거래정보' 26개가 주어진 DATA SET이며 아래는 간단한 요약 예시이다.

| 성별 || 연령구분 || 직업구분 || 지역구분 || 가구소득 || 결혼여부 || 맞벌이여부 || 자녀수 || 총자산 || 금융자산 || 부동산자산 |
|:---:|-|:---:|-|:---:|-|:---:|-|:---:|-|:---:|-|:---:|-|:---:|-|:---:|-|:---:|-|:---:|-|
|1    ||2    ||2    ||4    ||1    ||1    ||-    ||-    ||2,300||300  ||0    ||
|1    ||3    ||2    ||2    ||1    ||1    ||-    ||-    ||41,900||5,900||30,500||
|2    ||5    ||2    ||2    ||1    ||-    ||1    ||1    ||7,500||7,000||500  ||
|1    ||2    ||2    ||4    ||1    ||2    ||2    ||1    ||2,300||0    ||80   ||

위의 데이터를 살펴보면 좌측부터 8개는 '고객기본정보'로서 범주형자료이며 그 이후는 수치형자료이다. 
이와 같은 형태의 데이터가 17076개가 존재하며 범주들은 각각 2, 5, 9, 5, 7, 3, 3, 5의 level을 가지고 결혼여부, 맞벌이여부, 자녀수는 선택정보이기 때문에 결측값을 입력할 수 있게 구성되어있다. 

이를 토대로 141,750의 고객유형의 '금융거래정보'를 만드는 것이 첫 번째 과제이다. 다음의 과정으로 Test set를 만들 것이다.

### Imputation for missing data

#### Step 1 ) 군집화
동일한 기본정보 조합에 대해서 금융변수 값들의 편차가 존재한다. 이에 따라 금융정보가 유사한 집단을 만들 필요가 있으며, 추정하고자 하는 금융변수를 종속변수로 두고 그 외 금융변수들을 독립변수로하여 후진제거법을 실시한다. 그 후, 종속변수와 살아남은 독립변수들로 K-means로 군집화 해 추정하고자 하는 금융변수에 대해 유사한 집단을 만든다.

#### Step 2 ) 유사성 탐색
먼저, 우리가 만들 Test set는 Train data에 없는 고객유형이 존재한다. 

따라서 Train data의 고객유형으로 금융정보를 예측하는 것은 불가능하다고 판단되어 결측값이 포함된 기본정보를 제외하고 나머지 기본정보들과 유사한 조합을 고려한다.

![](/assets/SHB/step2.jpg)

그림을 보다시피 '고객기본정보'가 일치하는 것들을 빨간글씨로 표시해놓았고 이를 토대로 '금융거래정보'가 유사한 군집에서 '고객기본정보'가 제일 유사한 집단으로 금융거래정보를 예측할 것이다.

#### Step 3) Object matrix 생성
각 군집에서 a가 가장 큰 '고객기본정보' 유형으로 Object Matrix생성 후 missForest방식에 기반한 결측값 추정을 진행한다.

![](/assets/SHB/step3.jpg)

이때, 결측값 99는 Train set에 없는 조합이므로 missForest를 할때 해당 값이 포함된 열을 제거 후 추정하며 $\max$ a가 1개라면 '고객기본정보'유형의 금융정보를 그대로 가져오기 때문에 갖고있는 데이터에 의존하는 방식이다.

참고) missForest논문 : https://academic.oup.com/bioinformatics/article/28/1/112/219101

이 추정방식의 특징으로는 

* 유일한 기본정보 조합에 대해 금융변수 값을 그대로 사용함

* 군집을 미리 실시하고 missForest를 실시하므로 군집의 특징을 가지면서 추정시간을 단축

* 각 금융변수들을 개별적으로(pathwise) 추정하므로 전체를 추정하는 방식에 비해 성능이 좋음

을 말할 수 있다. 

## 2. Make peer group

위의 추정방식으로 만든 Test set으로 '금융거래정보'를 이용해 유사한 집단을 묶는 것이 두번째 과제이다. 이를 통해 본인의 정보를 입력하면 본인과 비슷한 사람들의 평균 금융거래정보를 보여주는 과정이다.

먼저, Max Webber's theory of Social Stratification이란 경제, 사회, 권력의 측면에서 각각 상, 중, 하류층으로 나뉘어 있으며 이들의 종합이 개인을 나타낸다고 보는 이론이다. 이에 착안하여 금융생활 또한 자산, 부채, 투자, 저축 분야로 나누어 군집분석을 실시해 이를 종합해 하나의 군집으로 보는 아이디어를 생각하였다. 

![](/assets/SHB/max_webber.jpg)

예를 들어, 각각 자산부분 1번, 부채부분 2번, 투자부분 3번, 저축부분 5번 군집의 경우 (1, 2, 3, 5)라는 군집으로 보는 것이다. 이 군집을 나누는 방법은 Self-Organizing-Map(SOM, 자기조직화지도)를 사용하였다. 

위의 Test set을 만들 때에는 14만개를 예측해야하기 때문에 차원이 낮고 속도의 중요성 때문에 K-means를 사용하였고, 이 부분에서는 속도가 중요한 것이 아니라 높은 차원을 다루는 것이 중요했기 때문에 SOM을 사용하였다.

간단히 SOM을 설명하자면 차원축소와 군집분석을 동시에 실시하는 알고리즘이다. 고차원의 데이터를 저차원(2차원)에서 보여주고 각각의 Map을 형성하는 인공신경망의 한 종류이다.

참고 ) SOM논문 : https://www.sciencedirect.com/science/article/pii/S0893608012002596

### Clustering by SOM

#### Step 1) 
각 분야별(자산, 저축, 투자, 부채)를 16개로 군집하여 조합해 16X16X16X16의 조합을 생성한다.

#### Step 2)
군집의 갯수가 많거나 군집내의 갯수가 적은 것은 의미가 없기 때문에 총 군집의 수를 조절하고자 각 군집에서 200개 이상인 부분만 채택한다.

#### Step 3)
이로 총 91개의 군집 조합이 재생성되고, 200개 미만인 군집의 조합은 없기 때문에 군집내 개체수가 200개 이상인 데이터를 Train data, 200개 미만인 데이터를 Test data로 하여 randomForest를 이용해 최종 91개의 군집을 만든다.

그 결과는 다음과 같이 표시될 수 있다. 

![](/assets/SHB/cluster.jpg)

## 3. Make quantile for peer group

위의 클러스터된 자료를 바탕으로 본인의 금융거래 정보를 입력하면 비슷한 사람들 중에 본인의 위치를 비교해 보여주는 시스템을 만드는 것이다. 

이를 위해선 100분위수를 주어 자신이 속한 quantile이 점수가 되는 방식이며 4가지로 분할해서 클러스터링을 한 것이기 때문에 고객의 금융생활 파악에 자세한 도움을 주고, 분류된 그룹의 특징들을 고려해 금융상품을 추천할 수 있다는 장점을 갖는다.

![](/assets/SHB/peer.jpg)

다음과 같이 고객이 본인의 정보를 입력하면 본인과 비슷한 사람들의 평균 금융거래정보를 보여주고 본인의 금융거래 정보를 입력하면 비슷한 사람들 중에 본인의 위치를 비교해 보여주는 시스템을 만들었다.

## 4. Drop basic information

지금까지 만든 자료들은 고객들의 설문조사를 바탕으로 만들어졌다. 하지만 입력해야하는 '고객기본정보'가 많아지면 cost가 좋지 않기 때문에 수집할 수 있는 정보는 적으면 적을수록 좋다. 따라서 8개의 항목 중 고객정보수집을 최소화하여 상담시스템을 만들 수 있다면 어떤 정보가 필요한지가 마지막 추가 과제이다. 

지금까지 만들어온 상담 시스템은 '고객기본정보'를 수집하여 금융생활의 위치를 파악하는데 목적을 두었다. 그에 따라 '금융거래정보'와 '고객기본정보'간의 상관성은 중요한 이슈이다. 이 때, 금융변수로 기본정보를 예측하는 모델을 만들었을 때 금융변수들로 기본정보를 잘 분류하지 못한다는 것은 전체 금융변수가 해당 기본정보를 잘 설명하지 못한다는 것을 의미한다. 위의 과정에서 만든 Test set의 금융변수는 기본정보의 특징을 잘 반영하고 있다. 

따라서 각각의 기본정보를 종속변수로 두고 금융정보들을 독립변수로 두었을 때 분류를 잘 하지 못한다면 기본정보는 금융변수들로 설명이 되지 않으므로 버릴 수 있다고 판단하였다.

### Drop by ranger

#### Step 1)  
8가지의 기본정보들을 각각 종속변수로 두고 금융변수들을 독립변수로 두는 모델을 세팅

#### Step 2)
ranger를 사용하여 각 군집별로 5-fold CV로 분류율을 계산한다. randomForest를 쓸 수도 있지만 5-fold에 91개의 군집과 8개의 기본정보를 이용하므로 연산량이 많아 메모리의 사용을 줄이기 위해 ranger를 사용한다. 

참고 ) Ranger : https://cran.r-project.org/web/packages/ranger/ranger.pdf

#### Step 3)
91X8의 분류율 matrix를 생성하고 각 기본정보에 대해 분류율을 바탕으로 귀무가설을 군집 분류율을 0.4로 두고 Wilcox test를 통해 검증한다.

| 변수| 성별 | 연령구분 | 직업구분 | 지역구분 | 가구소득 | 결혼여부 | 맞벌이여부 |자녀수| 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|p-value|1  |1    |1    |1    |1    |0.029|1    |1    |


검정 결과 결혼여부의 유의확률이 0.029로 유의수준 0.05하에서 분류율이 40%가 되지 않는 것을 확인하였다. 그리고 우리는 갖고있는 분류율 matrix의 모분포를 알 수 없기 때문에 비모수적 검정을 진행하였고 모수적 검정에 비해 3/$\pi$(약 97%)의 검정력을 갖고 있다.

## 5. R-code

### 5-1. Transform data

신한은행의 고객데이터이므로 이를 사용할 수 없으므로 간단한 예제 데이터로 보이겠다.

```r
rm(list = ls())
library(ISLR)
data <- Carseats;#attach(data)

# transform data type

# ShelveLoc
ShelveLoc.level <- c("Bad", "Medium", "Good")
for(i in 1:3){data$ShelveLoc<-ifelse(data$ShelveLoc==ShelveLoc.level[i],i,data$ShelveLoc)}

# AgeShelveLoc
Age.level <- c(30, 40, 50, 60)
data$Age <- ifelse(data$Age < 30, 1, data$Age)
for(i in 2:4){
  data$Age<-ifelse(Age.level[i-1] <= data$Age & data$Age < Age.level[i], i, data$Age)}
data$Age <- ifelse(data$Age >= 60, 5, data$Age)

# Education
Education.level <- 10:18
for(i in 1:9){data$Education<-ifelse(data$Education==Education.level[i],i,data$Education)}

# Urban
data$Urban <- factor(as.numeric(data$Urban)) # No is 1

# Us
data$US <- factor(as.numeric(data$US)) # No is 1
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

Age에 대해 20대, 30대, 40대, 50대, 60대 이상으로 구분하였으며 ShelveLoc, Age, Education, Urban, US는 범주형자료이다. 

5개의 범주형자료와 6개의 수치형자료가 혼합된 자료이며 이로 만들 수 있는 기본정보의 조합은 3X5X9X2X2 = 540개이고 이를 먼저 만들어보자

### 5-2. Find correlation of independent variables

먼저 추정해야할 변수(Sales)에 대해 연관성이 있는 다른 독립변수들(CompPrice, Income, Advertising, Population, Price)을 찾아보자.

```r
# scale for k-means and regression
m <- apply(data[, 1:6], 2, mean)
s <- apply(data[, 1:6], 2, sd)
data[, 1:6] <- scale(data[, 1:6])

l1 <- lm(data[, 1] ~ data[, 2] + data[, 3] + data[, 4] + data[, 5] + data[, 6])
b.l1 <- step(l1, direction = "backward", trace = F)
l2 <- lm(data[, 2] ~ data[, 1] + data[, 3] + data[, 4] + data[, 5] + data[, 6])
b.l2 <- step(l2, direction = "backward", trace = F)
l3 <- lm(data[, 3] ~ data[, 1] + data[, 2] + data[, 4] + data[, 5] + data[, 6])
b.l3 <- step(l3, direction = "backward", trace = F)
l4 <- lm(data[, 4] ~ data[, 1] + data[, 2] + data[, 3] + data[, 5] + data[, 6])
b.l4 <- step(l4, direction = "backward", trace = F)
l5 <- lm(data[, 5] ~ data[, 1] + data[, 2] + data[, 3] + data[, 4] + data[, 6])
b.l5 <- step(l5, direction = "backward", trace = F)
l6 <- lm(data[, 6] ~ data[, 1] + data[, 2] + data[, 3] + data[, 4] + data[, 5])
b.l6 <- step(l6, direction = "backward", trace = F)
```

그 결과 다음의 연관성을 확인할 수 있다. 이는 후에 k-means를 할 때 사용될 것이다.

* Sales는 ComPrice, Income, Advertising, Price (num : 1, 2, 3, 4, 6)
* ComPrice는 Sales, Income, Advertising, Population, Price (num : 1, 2, 3, 4, 5, 6)
* Income는 Sales, ComPrice, Price (num : 1, 2, 3, 6)
* Advertising는 Sales, ComPrice, Population, Price (num : 1, 2, 4, 5, 6)
* Population는 ComPrice, Advertising (num : 2, 4, 5)
* Price는 Sales, ComPrice, Income,  Population (num : 1, 2, 3, 4, 6)

이제 이를 바탕으로 k-means와 추정을 해야하는데 그 전에 기본적인 test set을 만들어보자.

```r
# null factor matrix
test_X <- as.data.frame(matrix(NA, 3*5*9*2*2, 5)) # essential for data.frame
test_X[,1] <- rep(1:3, each = 5*9*2*2, time = 1)
test_X[,2] <- rep(1:5, each = 9*2*2, time = 3)
test_X[,3] <- rep(1:9, each = 2*2, time = 5)
test_X[,4] <- rep(1:2, each = 2, time = 9)
test_X[,5] <- rep(1:2, each = 1, time = 2)

head(test_X)
```
```r
## V1 V2 V3 V4 V5
## 1 1 1 1 1 1
## 2 1 1 1 1 2
## 3 1 1 1 2 1
## 4 1 1 1 2 2
## 5 1 1 2 1 1
## 6 1 1 2 1 2
```

### 5-3. Make A-matrix

위에서 설명하였던 유사성을 설명하는 a matrix를 만들어보자. 
신한은행데이터를 위해선 Rcpp로 만들어야하나, 현재 데이터의 dimension이 그렇게 크지 않으므로 단순 반복문으로 Amatrix를 만든다.

```r
a.train <- data[, -(1:6)]
Amatrix <- matrix(NA, nrow(test_X), nrow(a.train))
for(i in 1:nrow(Amatrix)){
    for(j in 1:ncol(Amatrix)){                
        Amatrix[i, j] <- sum(test_X[i, ] == a.train[j, ])}}
head(Amatrix[,1:10])
```
```r
## [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
## [1,] 1 1 0 0 2 2 1 1 3 1
## [2,] 2 2 1 1 1 3 0 2 2 2
## [3,] 2 2 1 1 3 1 2 2 2 0
## [4,] 3 3 2 2 2 2 1 3 1 1
## [5,] 1 0 0 0 2 2 1 0 2 1
## [6,] 2 1 1 1 1 3 0 1 1 2
```

### 5-4 Simulation with K-means, A-matrix, missForest

```r
# set pred function
pred_function <- function(full.X, new.a, y.num, x.num, k.num = 4, new.x, sel.num){
  
  # transform numbering
  X.data <- full.X[,x.num] # full.X must be trans matrix
  Y.data <- full.X[,y.num] # X.data must be numeric data
  for(j in 1:ncol(Y.data)){Y.data[, j] <- factor(Y.data[, j])}
  # k-means
  k <- kmeans(X.data, centers = k.num)$cluster
  # testing
  test.k <- matrix(NA, 2, k.num); test.k <- rbind(test.k, 1:k.num)
  obj.data <- cbind(Y.data, X.data, k, new.a)
  # find similar base information
  for(z in 1:k.num){
    a <- obj.data[obj.data$k == z, ncol(obj.data)]
    test.k[1, z] <- max(a)
    test.k[2, z] <- sum(a == max(a))
  }
  
  max.1 <- max(test.k[1,])
  max.2 <- max(test.k[2,test.k[1,] == max.1])
  
  for(j in 1:ncol(test.k)){
    if(sum(test.k[1,j] == max.1, test.k[2,j] == max.2) == 2){
    best.k <- test.k[3, j];best.a <- test.k[1, j]}}
  # set training data
  obj.data <- obj.data[obj.data$k == best.k,]
  obj.data <- obj.data[obj.data[,ncol(obj.data)] == best.a, c(y.num, sel.num)]
  if(sum(obj.data[,length(y.num)+1]==0)==nrow(obj.data))obj.data[,length(y.num)+1]<-1e-10 
  # if data has a lot of 0, it can be error
  na.row <- unname(c(new.x))
  obj.data <- rbind(obj.data, na.row)
  
  mf <- missForest(obj.data)
  pred <- (mf$ximp[-(1:(nrow(obj.data)-1)), 6])*s[sel.num - 5] + m[sel.num - 5]
  if(pred < 0.1) pred <- 0 
  
  return(pred)
}
# simulation for setting
library(missForest)
train.data <- cbind(data[, -(1:6)], data[, 1:6])
test.data  <- cbind(test_X, NA, NA, NA, NA, NA, NA)

head(cbind(train.data[,1:5], round(train.data[,-(1:5)], 3)))
```
```r
## Shelve Age Ed Ur US Sales CompP In Ad Pop Price
## 1 1 3 8 2 2 0.709 0.849 0.155 0.656 0.076 0.178
## 2 2 5 1 2 2 1.319 -0.911 -0.738 1.408 -0.033 -1.385
## 3 3 4 3 2 2 0.908 -0.781 -1.203 0.506 0.028 -1.512
## 4 3 4 5 2 2 -0.034 -0.520 1.120 -0.396 1.365 -0.794
## 5 1 2 4 2 1 -1.185 1.045 -0.166 -0.547 0.510 0.515
## 6 1 5 7 1 2 1.173 -0.064 1.584 0.957 1.602 -1.850
```

```r
# simulation for Sales
k_num <- 3
sel_num <- 6 # Sales
y_num <- 1:5
x_num <- c(1, 2, 3, 4, 6) + 5 # calculated by regression

for(i in 1:540){
  a.matrix <- as.matrix(Amatrix[i,])
  test_X[i, 6]<-(pred_function(full.X=train.data,new.a=a.matrix,y.num=y_num,x.num=x_num,
                               k.num=k_num,new.x=test.data[i,],sel.num=sel_num))
  cat(paste("iter",i,"=",test_X[i,6]),"\n")
}

# simulation for CompPrice
k_num <- 3
sel_num <- 7 # CompPrice
y_num <- 1:5
x_num <- c(1, 2, 3, 4, 5, 6) + 5 # calculated by regression

for(i in 1:540){
  a.matrix <- as.matrix(Amatrix[i,])
  test_X[i, 7]<-(pred_function(full.X=train.data,new.a=a.matrix,y.num=y_num,x.num=x_num,
                               k.num=k_num,new.x=test.data[i,],sel.num=sel_num))
  cat(paste("iter",i,"=",test_X[i,7]),"\n")
}

# simulation for Income
k_num <- 3
sel_num <- 8 # Income
y_num <- 1:5
x_num <- c(1, 2, 3, 6) + 5 # calculated by regression

for(i in 1:540){
  a.matrix <- as.matrix(Amatrix[i,])
  test_X[i, 8]<-(pred_function(full.X=train.data,new.a=a.matrix,y.num=y_num,x.num=x_num,
                               k.num=k_num,new.x=test.data[i,],sel.num=sel_num))
  cat(paste("iter",i,"=",test_X[i,8]),"\n")
}

# simulation for Advertising
k_num <- 3
sel_num <- 9 # Advertising
y_num <- 1:5
x_num <- c(1, 2, 4, 5, 6) + 5 # calculated by regression

for(i in 1:540){
  a.matrix <- as.matrix(Amatrix[i,])
  test_X[i, 9]<-(pred_function(full.X=train.data,new.a=a.matrix,y.num=y_num,x.num=x_num,
                               k.num=k_num,new.x=test.data[i,],sel.num=sel_num))
  cat(paste("iter",i,"=",test_X[i,9]),"\n")
}

# simulation for Population
k_num <- 3
sel_num <- 10 # Population
y_num <- 1:5
x_num <- c(2, 4, 5) + 5 # calculated by regression

for(i in 1:540){
  a.matrix <- as.matrix(Amatrix[i,])
  test_X[i, 10]<-(pred_function(full.X=train.data,new.a=a.matrix,y.num=y_num,x.num=x_num,
                                k.num=k_num,new.x=test.data[i,],sel.num=sel_num))
  cat(paste("iter",i,"=",test_X[i,10]),"\n")
}

# simulation for Price
k_num <- 3
sel_num <- 11 # Price
y_num <- 1:5
x_num <- c(1, 2, 3, 4, 6) + 5 # calculated by regression

for(i in 1:540){
  a.matrix <- as.matrix(Amatrix[i,])
  test_X[i, 11]<-(pred_function(full.X=train.data,new.a=a.matrix,y.num=y_num,x.num=x_num,
                                k.num=k_num,new.x=test.data[i,],sel.num=sel_num))
  cat(paste("iter",i,"=",test_X[i,11]),"\n")
}

test.data <- test_X
colnames(test.data) <- colnames(data)
head(cbind(test.data[,1:5], round(test.data[,-(1:5)], 3)))
```
```r
## Sales CompP In Ad Pop Price Shelve Age Ed Ur US
## 1 1 1 1 1 1 8.190 115.800 93.120 1.182 854.346 186.825
## 2 1 1 1 1 2 9.010 121.000 78.000 1.453 1001.722 163.148
## 3 1 1 1 2 1 5.323 137.155 54.854 3.187 854.346 139.472
## 4 1 1 1 2 2 11.670 125.000 89.000 11.832 1001.722 163.148
## 5 1 1 2 1 1 5.400 149.000 71.040 6.172 854.346 163.148
## 6 1 1 2 1 2 5.400 149.000 73.000 11.877 1001.722 163.148
```

### 5-5. Clustering by SOM

```r
set.seed(1) #essential
# data setting
library(kohonen)
raw.data <- test.data
test.data1 <- raw.data[,6]
test.data2 <- raw.data[,7]
test.data3 <- raw.data[,8]
test.data4 <- raw.data[,9]
test.data5 <- raw.data[,10]
test.data6 <- raw.data[,11]

best.som1<-som(X=scale(test.data1),grid=somgrid(xdim=2,ydim=1,topo="hexagonal"))
best.som2<-som(X=scale(test.data2),grid=somgrid(xdim=2,ydim=1,topo="hexagonal"))
best.som3<-som(X=scale(test.data3),grid=somgrid(xdim=2,ydim=1,topo="hexagonal"))
best.som4<-som(X=scale(test.data4),grid=somgrid(xdim=2,ydim=1,topo="hexagonal"))
best.som5<-som(X=scale(test.data5),grid=somgrid(xdim=2,ydim=1,topo="hexagonal"))
best.som6<-som(X=scale(test.data6),grid=somgrid(xdim=2,ydim=1,topo="hexagonal"))

dataset <- cbind(raw.data, clust1 = NA, clust2 = NA, clust3 = NA, 
                 clust4 = NA, clust5 = NA, clust6 = NA, clust = NA)
dataset$clust1<- best.som1$unit.classif
dataset$clust2<- best.som2$unit.classif
dataset$clust3<- best.som3$unit.classif
dataset$clust4<- best.som4$unit.classif
dataset$clust5<- best.som5$unit.classif
dataset$clust6<- best.som6$unit.classif
dataset$clust <- paste(dataset$clust1, dataset$clust2, dataset$clust3, 
                       dataset$clust4, dataset$clust5, dataset$clust6, sep=",")

clust<-as.data.frame(table(dataset$clust)) 
clfreq<-clust[clust$Freq >= 10,]

for(i in 1:nrow(clfreq)){
  dataset$clust<-ifelse(dataset$clust==clfreq[i, 1],i,dataset$clust)}

dataset$clust <- as.numeric(dataset$clust)
train.clust<- dataset[is.na(dataset$clust) == F,]
train.clust[,18] <- factor(train.clust[,18])
test.clust <- dataset[is.na(dataset$clust) == T,]
```

많은 수의 군집은 의미가 없고 군집내의 수가 적은 것 또한 의미가 없다. 또한 결과는 Warning이 나오지만 관측값의 수가 10이상인 것만 군집으로 볼 것이기 때문에 paste로 결합된 군집들은 numeric으로 바꾸어주면서 NA로 바뀌게 된다. 이러한 tricky한 방법으로 결측치를 만들어주고 결측치가 포함된 관측값들을 Test set으로 만들고 결측치가 포함되지 않은 관측값들로 Train set으로 훈련시켜, 군집을 예측한다. 

```r
library(randomForest)

rf.fit <- randomForest(train.clust[,18] ~., train.clust[,6:11], method = "class")
test.clust[,18] <- predict(rf.fit, test.clust[,6:11], type = "class")
clust.data <- rbind(train.clust, test.clust)
head(clust.data, 4)
```
```r
## Sales CompP In Ad Pop Price Shelve Age Ed Ur
## 3 1 1 1 2 1 5.323233 137.1550 54.85417 3.1866214 854.3457
## 5 1 1 2 1 1 5.400000 149.0000 71.04000 6.1720173 854.3457
## 6 1 1 2 1 2 5.400000 149.0000 73.00000 11.8767219 1001.7222
## 7 1 1 2 2 1 4.014827 125.8265 55.57250 0.1587629 854.3457
## US clust1 clust2 clust3 clust4 clust5 clust6 clust
## 3 139.4717 1 1 2 1 2 1 7
## 5 163.1483 1 1 1 1 2 1 2
## 6 163.1483 1 1 1 2 1 1 4
## 7 139.4717 1 1 2 1 2 1 7
```

### 5-6. Peer group

```r
clust.level <- 1:length(table(clust.data[,ncol(clust.data)]))
sel.data <- clust.data[,6:11]
peer.matrix <- matrix(NA, 6*length(clust.level), 99)

for(i in 1:length(clust.level)){
  
  j <- 6*(i-1)+1
  peer.matrix[j,] <- round(quantile(sel.data[clust.level==i,1],seq(0.01,0.99,0.01)),0)
  peer.matrix[j+1,]<-round(quantile(sel.data[clust.level==i,2],seq(0.01,0.99,0.01)),0)
  peer.matrix[j+2,]<-round(quantile(sel.data[clust.level==i,3],seq(0.01,0.99,0.01)),0)
  peer.matrix[j+3,]<-round(quantile(sel.data[clust.level==i,4],seq(0.01,0.99,0.01)),0)
  peer.matrix[j+4,]<-round(quantile(sel.data[clust.level==i,5],seq(0.01,0.99,0.01)),0)
  peer.matrix[j+5,]<-round(quantile(sel.data[clust.level==i,6],seq(0.01,0.99,0.01)),0)  
}
colnames(peer.matrix) <- sprintf("%s", paste0("percentile", 1:99))
peer.num <- rep(1:length(clust.level), each = 3)
peer.matrix <- cbind(peer.num, peer.matrix)
head(peer.matrix[,1:5])
```

데이터의 수에 비해 quantile의 수가 많아 겹치는 부분이 많이 생겼으나 데이터의 양이 많아지면 잘 구분할 수 있을 것이다. 끝으로 변수제거를 보고 마치겠다.

### 5-7. Drop variable

```r
#library(randomForest)
library(ranger) # faster than randomForest
set.seed(1)

for(j in 1:5){clust.data[,j] <- factor(clust.data[,j])}
f.cv.err <- matrix(NA, length(clust.level), 5)

for(col in 1:length(clust.level)){ # col is clust level
  
  k <- 3
  cor.data <- clust.data[clust.data[,ncol(clust.data)]==col,-ncol(clust.data)]
  r <- sample(nrow(cor.data), (nrow(cor.data)%/%k)*k, replace = F)
  temp.data <- cor.data[r,]
  cv.err <- rep(NA, 5)
  
  for(j in 1:5){ # j is class variable
    
    mse <- rep(NA, k)
    
    for(i in 1:k){ # i is cv index
      
      test <- ((i-1)*(nrow(temp.data)/k)+1):(i*(nrow(temp.data)/k))
      test.data <- temp.data[test,]
      train.data <- temp.data[-test,]
      
      train.y <- train.data[,j] 
      train.X <- train.data[,-(1:5)]
      
      test.y <- test.data[,j]
      test.X <- test.data[,-(1:5)]
      
      #rf.fit <- randomForest(train.y~., data = train.X, method = "class")
      #pred <- predict(rf.fit, test.X, type = "class")
      rf.fit <- ranger(train.y~., data = train.X, classification = T)
      pred <- predict(rf.fit, test.X)$predictions
      tab <- table(test.y, pred)
      
      mse[i] <- sum(diag(tab))/sum(tab)
      
      #cat(paste("col_iter",col, "j_iter", j, "i_iter",i,"=",mse[i]),"\n")
    }
    cv.err[j] <- mean(mse)
  }
  f.cv.err[col,] <- cv.err
}
w.p.value<-matrix(NA, 1, ncol(f.cv.err));colnames(w.p.value)<-colnames(f.cv.err)
for(j in 1:length(w.p.value)){
  w.p.value[j]<-wilcox.test(na.omit(f.cv.err[,j]),mu=0.2,alternative="less")$p.value} 

colnames(f.cv.err) <- paste0(colnames(clust.data)[1:5], "_err")
result <- apply(na.omit(f.cv.err), 2, mean)
result
```
```r
## Sales_err CompP_err In_err Ad_err Pop_err
## 0.58217968 0.29407843 0.09216345 0.57467307 0.64934915
```
```r
w.p.value
```
```r
## [,1] [,2] [,3] [,4] [,5]
## [1,] 0.9999684 0.9969692 0.0001654601 0.9999813 0.9999396
```

제거해도 괜찮은 변수는 Income 범주로 나타났다.

## Conclusion

이로서 2018 신한은행 빅콘테스트에서 최우수상 - 한국정보통진흥협회상을 받은 프로젝트를 리뷰해보았다. 12팀이 올라온 2차 발표에서 피드백을 받았던 것은 전체적으로 문제될 게 없었으며 군집부분은 굉장히 좋은 아이디어고 이를 활용할 수 있는 방안까지 제시했다는 점에서 높은 평가를 받았다. 

그러나 결측치를 채운 값들이 다소 안좋은 결과를 가져왔다고 하시며 군집의 수를 정하는 데에 문제가 있었다라고 말씀하셨다. 

나의 생각은 조금 다르다. 14만여개의 기본정보 조합의 금융정보를 예측하는 데에 있어 1만 7천여개의 설문조사자료는 엄청난 정보를 갖고 있다고 초기에 판단하였다. 신한은행 측에서는 결측치를 추정한 참값을 갖고 있다고 했고 그와 비교했다고 하였다. 나는 그러한 자료를 갖고 있을 것이라는 정보도 없었고 생각하지 못해 주어진 1만 7천여개의 설문조사 값에 최대한 의존해야한다고 판단했다. 예를 들어 20대, 남성, 기혼자, ... 의 기본정보 조합이 만약 유일하다면 나의 추정 방식은 그 조합의 금융정보를 그대로 갖고 오는 알고리즘이다. 이는 참 값은 없을 것이라는 가정하에 기존 데이터에 최대한 의존하는 방식이다.

하지만 이는 동시에 위험함을 초래할 수도 있는데, 새로운 관측값 하나만 등장하더라도 편차가 굉장히 클 가능성이 매우 높다는 것이다. 그런 점에서 조금은 아쉬움이 남지만, 머신러닝 기법을 사용해 실데이터를 다루어 보고 그에 따른 성과를 얻었다는 점에서 굉장히 좋은 경험이었다. 

머신러닝을 이용한 성과를 얻었으니, 다음은 딥러닝을 이용해 논문을 쓰고 또 다른 프로젝트에서 활용하는 것을 목표로 하고 있다.

## Reference

Daniel J.Stekhoven and Peter Buhlmann (2011) **MissForest** : *non-parametric missing value imputation for mixed-type data*. Department of Mathematics, ETH Zurich

Fernando Bacao, Victor Lobo, and Marco Painho (2005) *Self-organizing Maps as Substitutes for K-Means Clustering, ICCS 2005, LNCS 3516, pp. 476 ~ 483, 2005.*

Marvin N.Wright and Andreas Ziegler (2017) **ranger** *A Fast Implementation of Random Forests for High Dimensional Data in C++ and R* . Universitat zu Lubeck

김우철(2012) *수리통계학, 민음사*

Knuth DE (1995) *The Art of Computer Programming volume 2, Addison-Wesley*
