---
layout: post
title:  "Application Lecture3"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. Review

## 1-1. 정의 및 목적

- $X$(독립변수, 원인)와 $Y$(종속변수, 범주형 자료) 사이의 관계를 표현한 식

- 새로운 관측치가 왔을 때 이를 기존 범주 중 하나로 예측

- 예시

    + 제품이 불량인지 양품인지 분류
    
    + 고객이 이탈고객인지 잔류고객인지 분류
    
    + 카드 거래가 정상인지 사기인지 분류
    
    + 이메일이 스팸인지 정상인지 분류
    
- $Y$(종속변수)의 형태가 예($Y = 1$) 혹은 아니오($Y = 0$)으로 나뉨

- $Y$가 1을 가질 확률을 만들어주는 모형

$$
\begin{split}
&E(Y) = p = P(Y = 1) = \frac{1}{1+ \exp(-\beta_0 + \beta_1 X)}\\
&\Rightarrow \log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X
\end{split}
$$

```r
rm(list = ls())
library(ggplot2)
data(iris)
iris$Species <- ifelse(iris$Species=="virginica", "virginica", "no virginica")
iris$Species <- factor(iris$Species)
p <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width))
s <- geom_point(aes(col = Species), size = 2)
l <- labs(title = "Adjusted iris data")
p + s + l
```

<center><img src="/assets/Application_lecture3/1.png"></center>

위의 분류를 가능하게 하는 최적의 분류 직선을 찾는 모델이 로지스틱 회귀분석모델이다.

로지스틱 회귀분석은 3단계에 걸쳐 진행된다.

1. 훈련데이터(train data) 예측데이터(test data)로 분할

2. 훈련데이터(train data)로 모델링

3. 예측데이터(test data)의 성능평가

## 1-2. Split train data, test data

먼저 훈련데이터와 예측데이터를 7:3의 비율로 분할한다.
```r
set.seed(1)
n <- nrow(iris) 
index <- sample(n, 0.7*n, replace = F)
train.data<- iris[index,]
test.data <- iris[-index,]
dim(train.data)
```
```r
## [1] 105   5
```
```r
dim(test.data)
```
```r
## [1] 45  5
```

## 1-3. Fitting train data

훈련데이터(train data)로 모델링을 진행한다.
```r
fit <- glm(Species ~ Sepal.Length + Sepal.Width, data = train.data, family = binomial)
```

## 1-4. Test data with accuracy

훈련데이터를 통해 만들어진 모델을 예측해보자.
```r
pred <- predict(fit, new = test.data, type = "response")
pred
```
```r
##           3           4           5           8           9          11 
## 0.008994761 0.007746129 0.012466142 0.015864005 0.005741925 0.032213155 
##          16          27          30          36          41          46 
## 0.031084190 0.015864005 0.008994761 0.020169099 0.014064280 0.014999212 
##          47          49          50          52          54          55 
## 0.012821496 0.024708557 0.017889852 0.484521970 0.194927305 0.668184210 
##          56          57          62          63          66          67 
## 0.184882164 0.387668508 0.234657296 0.517164343 0.706647884 0.119093058 
##          69          72          80          82          90          96 
## 0.648985729 0.403280443 0.224587289 0.176455712 0.159387971 0.150831148 
##          97          99         103         107         109         116 
## 0.167164813 0.059828065 0.890240432 0.035554473 0.833773157 0.484521970 
##         117         122         125         128         133         137 
## 0.611946320 0.147221227 0.653548929 0.346083840 0.605165239 0.359078790 
##         138         139         141 
## 0.515072836 0.287153778 0.706647884
```

변수 `pred`의 형태는 0부터 1사이의 확률이다. 

이를 범주화 시키기 위해 0.5이상을 `verginica`, 0.5미만을 `no verginica`로 잡는다.

그 후 모델의 성능을 평가하기 위한 척도로 
$$
Accuracy = \frac{1}{n}\sum_{i = 1}^nI(y_i == \hat{y}_i)
$$

를 사용하며 그 값은 다음과 같다.
```r
y.hat <- ifelse(pred > 0.5, "virginica", "no virginica")
y.hat
```
```r
##              3              4              5              8              9 
## "no virginica" "no virginica" "no virginica" "no virginica" "no virginica" 
##             11             16             27             30             36 
## "no virginica" "no virginica" "no virginica" "no virginica" "no virginica" 
##             41             46             47             49             50 
## "no virginica" "no virginica" "no virginica" "no virginica" "no virginica" 
##             52             54             55             56             57 
## "no virginica" "no virginica"    "virginica" "no virginica" "no virginica" 
##             62             63             66             67             69 
## "no virginica"    "virginica"    "virginica" "no virginica"    "virginica" 
##             72             80             82             90             96 
## "no virginica" "no virginica" "no virginica" "no virginica" "no virginica" 
##             97             99            103            107            109 
## "no virginica" "no virginica"    "virginica" "no virginica"    "virginica" 
##            116            117            122            125            128 
## "no virginica"    "virginica" "no virginica"    "virginica" "no virginica" 
##            133            137            138            139            141 
##    "virginica" "no virginica"    "virginica" "no virginica"    "virginica"
```
```r
accuracy <- mean(test.data$Species == y.hat)
accuracy
```
```r
## [1] 0.7777778
```

시각화를 진행해보도록 하자. 그러기 위해 `grid`를 정의한다.
```r
grid1 <- seq(min(iris$Sepal.Length), max(iris$Sepal.Length), length.out = 100)
grid2 <- seq(min(iris$Sepal.Width), max(iris$Sepal.Width), length.out = 100)
grid <- expand.grid(Sepal.Length = grid1, Sepal.Width = grid2)
pred.grid <- ifelse(predict(fit, new = grid, type= "response")>=0.5, "virginica", "no virginica")
pred.grid <- factor(pred.grid)
z <- data.frame(grid1 = grid[,1], grid2 = grid[,2], pred.grid)
p1 <- geom_point(data = z, aes(x = grid1, y = grid2, col = pred.grid), alpha = 0.1)
l1 <- labs(title = "Decision boundary iris data")
p + s + p1 + l1
```

<center><img src="/assets/Application_lecture3/2.png"></center>

# 2. Apply real data

## 2-1. Import crime data

먼저 `crime.csv`데이터를 불러와보자.
```r
rm(list = ls())
library(ggplot2)
library(reshape)
library(leaflet)
crime <- read.csv(file.choose())
summary(crime)
```
```r
##            Category    DayOfWeek    PdDistrict         X         
##  MISSING PERSON:6200   Fri:1847   A      :2319   Min.   :-122.5  
##  ROBBERY       :7000   Mon:2096   C      :1784   1st Qu.:-122.4  
##                        Sat:1876   H      :1652   Median :-122.4  
##                        Sun:1826   D      :1614   Mean   :-122.4  
##                        Thu:1743   I      :1422   3rd Qu.:-122.4  
##                        Tue:1843   E      :1106   Max.   :-122.4  
##                        Wed:1969   (Other):3303                   
##        Y              dist            time      
##  Min.   :37.71   Min.   : 1.99   Min.   :13.68  
##  1st Qu.:37.73   1st Qu.: 8.31   1st Qu.:20.97  
##  Median :37.76   Median :15.90   Median :34.49  
##  Mean   :37.76   Mean   :14.15   Mean   :30.08  
##  3rd Qu.:37.78   3rd Qu.:19.59   3rd Qu.:38.47  
##  Max.   :37.81   Max.   :28.10   Max.   :47.68  
## 
```

## 2-2. Modeling

먼저 `crime`데이터를 훈련데이터(train data)와 예측데이터(test data)로 분할해준다.
```r
set.seed(1)
n <- nrow(crime) 
index <- sample(n, 0.7*n, replace = F)
train.data<- crime[index,]
test.data <- crime[-index,]
dim(train.data)
```
```r
## [1] 9240    7
```
```r
dim(test.data)
```
```r
## [1] 3960    7
```

그 다음, 훈련데이터를 이용해 로지스틱 회귀모형을 적합시킨다.
```r
fit <- glm(Category ~ DayOfWeek + PdDistrict + dist + time, data = train.data, family = binomial)
pred <- predict(fit, new = test.data, type = "response")
head(pred)
```
```r
##         1        11        12        13        17        20 
## 0.7343405 0.7480136 0.6836864 0.6793108 0.7389880 0.7914620
```

확률값을 범주로 바꿔주기위해 아래의 코드를 사용한다.
```r
y.hat <- ifelse(pred > 0.5, "ROBBERY", "MISSING PERSON")
head(y.hat)
```
```r
##         1        11        12        13        17        20 
## "ROBBERY" "ROBBERY" "ROBBERY" "ROBBERY" "ROBBERY" "ROBBERY"
```
```r
accuracy <- mean(test.data$Category == y.hat)
accuracy
```
```r
## [1] 0.7681818
```

약 77%의 예측률을 갖고 있는 모델을 만들었다. 시각화를 진행해야하지만 우리는 2개의 변수를 사용한 것이 아니라, 여러개의 변수를 사용했기 때문에 시각화가 불가능하다.

따라서 성능평가와 실제 값과의 차이를 알아보는 시각화만을 진행한다.

그러기 위해 우선 예측 값의 속성을 알아보자.
```r
label <- rep(NA, nrow(test.data))
label <- ifelse(test.data$Category == "MISSING PERSON" & test.data$Category == y.hat,
                "Correct MISS",
                ifelse(test.data$Category == "MISSING PERSON" & test.data$Category != y.hat,
                       "False MISS",
                       ifelse(test.data$Category == "ROBBERY" & test.data$Category == y.hat,
                              "Correct ROBB", "False ROBB")))
table(label)
```
```r
## label
## Correct MISS Correct ROBB   False MISS   False ROBB 
##         1441         1601          455          463
```

`MISSING PERSON`을 맞춘 값과 틀린 값, `ROBBERY`를 맞춘 값과 틀린 값을 계산한 결과이다.

이를 통해 시각화를 진행한다.
```r
df <- cbind(label = label, pred = pred, test.data)
head(df)
```
```r
##           label      pred       Category DayOfWeek PdDistrict         X
## 1  Correct ROBB 0.7343405        ROBBERY       Thu          H -122.4079
## 11 Correct ROBB 0.7480136        ROBBERY       Thu          E -122.4327
## 12   False MISS 0.6836864 MISSING PERSON       Tue          H -122.4081
## 13   False MISS 0.6793108 MISSING PERSON       Sat          H -122.3952
## 17   False MISS 0.7389880 MISSING PERSON       Mon          B -122.4125
## 20   False MISS 0.7914620 MISSING PERSON       Tue          D -122.4197
##           Y     dist     time
## 1  37.78151 17.75811 36.78203
## 11 37.78329 16.77426 38.63680
## 12 37.78399 14.39583 37.95776
## 13 37.78924 17.36470 35.62333
## 17 37.80193 22.53707 34.72750
## 20 37.76505 20.61063 38.65506
```
```r
pal <- colorFactor(c("navy", "red", "purple", "orange"), domain = unique(df$label))

m <- leaflet() %>%
  addTiles() %>%
  setView(lng=-122.43317, lat=37.76345, zoom = 12)

a <- m %>% 
  addCircles(data = df, lng= ~X, lat= ~Y, 
             opacity = 2, weight = 4, color = ~pal(label)) %>%
  addLegend(data = df, pal = pal, values = ~label,
            opacity = 0.7, title = "Predict plot")
a
```
```r
<center><img src="/assets/Application_lecture3/3.png"></center>
```

`MISSING PERSON`을 틀린 값은 `ROBBERY`가 많았던 지역에 존재하는 것을 확인할 수 있다.

# 3. Question

다음의 형식을 이용해 `crime.csv`데이터를 이용해 분석 보고서를 완성하시오.

```r
---
title: "Classfication report via crime data"
author: "SSWU, Your Name"
date: \today
output: html_document
---
```
```r
` ```{r, echo = T, out.width = '70%', fig.align='center'}
` (your script)
` ```
```
### 1. EDA data
#### 1-1. Target variable
#### 1-2. Numeric variable
#### 1-3. Factor variable
### 2. Classification
#### 2-1. Modeling
#### 2-2. Test modeling
### 3. Conclusion
