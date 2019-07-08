---
layout: post
title:  "Application Lecture1"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 0. Lecture intro

- 신우영, sinwy93@korea.ac.kr

- 50분 강의, 10분 휴식, 50분 강의, 10분 휴식, 50분 문서화

- 수업내용은 유동적으로 변경될 수 있음

- 스케줄

|단계 | 주요내용 | 소요시간|
|:--:|:--:|:--:|
|1일차|R-markdown을 이용한 분석 보고서 만들기|3시간|
|2일차|빅 데이터를 이용한 데이터 고급 시각화 분석 보고서 만들기|3시간|
|3일차|빅 데이터를 이용한 로지스틱 분석 보고서 만들기|3시간|
|4일차|빅 데이터를 이용한 군집분석 보고서 만들기|3시간|
|5일차|텍스트 마이닝, 크롤링을 이용해 분석 보고서 만들기 |3시간|
|6일차|빅 데이터를 이용한 최종 분석보고서 만들기|3시간|

# 1. R-markdown

## 1-1. What is R-markdown

- 텍스트와 코드 및 사진들을 조합하여 형식화된 출력물을 생성할 수 있는 도구

- `R`, `python`, `SQL`을 비롯한 여러 언어를 사용할 수 있음 

- `HTML`, `PPT`, `PDF`을 비롯한 여러 문서형식을 생산할 수 있음

<center><img src="/assets/Application_lecture1/1.png"></center>

## 1-2. How to use R-markdown

먼저 `Lecture1`이라는 폴더를 바탕화면에 생성한다.

### 1-2-1. Make R-markdown form

<center><img src="/assets/Application_lecture1/2.PNG"></center>

<center><img src="/assets/Application_lecture1/3.PNG"></center>

```r
---
title: "Untitled"
author: "KU, Shin wooyoung"
date: "2019년 6월 2일"
output: html_document
---
```

이 부분으로 `R-markdown`의 형식을 구성한다.

- `title` : 문서의 전체 제목을 입력하는 부분

- `author` : 문서를 작성하는 작성자의 이름을 입력하는 부분 

- `date` : 문서의 작성 날짜를 입력하는 부분

- `output` : 문서의 형식을 결정하는 부분

    + `html_document`
  
    + `pdf_document`
  
    + `beamer_document`
  
    + 이번 강의에서는 한글이랑 호환이 잘 되는 `HTML`형식을 사용
    
    
문서를 만들기 위한(실행시키기 위한) 단축키는 `ctrl` + `shift` + `k` 이다. 이를 실행하면

<center><img src="/assets/Application_lecture1/4.PNG"></center>

이러한 결과를 확인할 수 있는데, 이는 `R`은 영어를 선호하지만 `date: "2019년 6월 2일"` 부분에 한글이 들어있기 때문에 발생하는 결과이다.

따라서 이를 실행하지 않고, `date: "2019년 6월 2일"` 부분을 `date: \today`로 변경해준다.

이는 오늘의 날짜를 입력하는 코드행이며 `example`이라는 이름으로 문서를 생성해본다.

<center><img src="/assets/Application_lecture1/5.PNG"></center>

이러한 문서를 확인할 수 있다. 

### 1-2-2. Input text

기본적인 텍스트(문자) 입력 방식은 일치한다. 

기본적으로 `enter`키를 이용해 줄 간격을 한 칸만 띄우는 것을 `R-markdown`에서는 인식하지 못한다.

따라서 한 줄을 띄우게 입력을 하고 싶다면 `enter`키를 2번 입력해야한다.

그리고 텍스트를 강조하는 방법은 여러가지가 존재한다.

* 안녕하세요 (기본)

* `안녕하세요` (' 사용)

* **안녕하세요** (** 사용)

* *안녕하세요* (* 사용)

그 다음은 문서의 소제목을 입력하는 방법이다. `#`의 갯수를 이용해 제목의 크기를 결정한다.

# 1st Level Header
## 2nd Level Header
### 3rd Level Header

주의해야할 점은 일반 텍스트와 소제목의 줄간격을 없이 입력하게 되면 문제가 발생한다

`안녕하세요`
### 3rd Level Header

따라서 간격을 주어서 입력해야 함을 주의하자.

다음은 항목을 열거하는 방법이다. `숫자`, `-`, `+`, `*`을 이용한다.

`숫자`를 이용하면 차례대로 숫자가 나오게 된다.

1. item1

2. item2

`-`, `+`, `*` 을 이용하면 숫자가 표시되지 않고 항목이 열거된다.

* item1

* item2

항목을 입력할 때도, 마찬가지로 `enter`를 2번 입력하는 것을 권장한다.

여기에 더불어 하위 항목을 붙이고 싶다면 아래와 같이 입력한다.

<center><img src="/assets/Application_lecture1/6.PNG"></center>

1. item1

    + sub item1
    
    + sub item2

2. item2

`enter`키를 2번 입력하고 `tab`키를 2번 입력해서 띄운다.

### 1-2-3. Input R-code

`R-code`를 입력하고 싶다면, 다음의 형식을 사용하여야한다.

```{r}
```{r}
rm(list = ls())
data(cars)
summary(cars)
` ```
```

* `r` : 내가 입력할 언어가 `R`이라는 것을 명시해주는 부분

```r
rm(list = ls()) 
data(cars) 
summary(cars) 
```
```r
##      speed           dist       
##  Min.   : 4.0   Min.   :  2.00  
##  1st Qu.:12.0   1st Qu.: 26.00  
##  Median :15.0   Median : 36.00  
##  Mean   :15.4   Mean   : 42.98  
##  3rd Qu.:19.0   3rd Qu.: 56.00  
##  Max.   :25.0   Max.   :120.00
```

그러나 나의 코드를 보여주고 싶지 않을 상황도 존재한다. 그럴 경우에는 다음의 코드를 사용한다.
```r
```{r, echo = F}
rm(list = ls())
data(cars)
summary(cars)
` ```
```

* `echo` : 코드의 출력을 담당하는 부분

```r
##      speed           dist       
##  Min.   : 4.0   Min.   :  2.00  
##  1st Qu.:12.0   1st Qu.: 26.00  
##  Median :15.0   Median : 36.00  
##  Mean   :15.4   Mean   : 42.98  
##  3rd Qu.:19.0   3rd Qu.: 56.00  
##  Max.   :25.0   Max.   :120.00
```

### 1-2-4. Input graph

그래프를 입력하거나, 외부 사진을 넣는 방법은 아래의 코드를 활용한다.

```r
```{r, out.width = '70%', fig.align='center'}
knitr::include_graphics("iris.png")
` ```
```

* `out.width` : 출력물의 크기를 조정하는 부분

* `fig.align` : 출력물의 위치를 조정하는 부분

주의해야할 점은 사진의 위치가 내가 작업하는 공간과 같은 공간에 있어야 한다.

<center><img src="/assets/Application_lecture1/iris.png"></center>

또한 `R-code`를 입력하여 그래프를 그릴 수도 있다.

```r
```{r, echo=FALSE}
plot(pressure)
` ```
```
```r
data(pressure)
plot(pressure, main = "pressure data plot") 
```

<center><img src="/assets/Application_lecture1/8.png"></center>

# 2. Exercise data

분석보고서를 만드는 데 필요한 `mtcars`데이터와 `iris`데이터를 이용해 회귀분석 및 로지스틱 회귀분석을 복습해보자.

## 2-1. Regression

회귀분석은 3단계로 이루어지며 

1. 원인과 결과의 설정

2. trend를 잘 설명할 수 있는 모델링

3. 만든 모델로 예측

이며 데이터의 형태를 먼저 살펴보자.

```r
data(mtcars)
str(mtcars)
```
```r
## 'data.frame':    32 obs. of  11 variables:
##  $ mpg : num  21 21 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 ...
##  $ cyl : num  6 6 4 6 8 6 8 4 4 6 ...
##  $ disp: num  160 160 108 258 360 ...
##  $ hp  : num  110 110 93 110 175 105 245 62 95 123 ...
##  $ drat: num  3.9 3.9 3.85 3.08 3.15 2.76 3.21 3.69 3.92 3.92 ...
##  $ wt  : num  2.62 2.88 2.32 3.21 3.44 ...
##  $ qsec: num  16.5 17 18.6 19.4 17 ...
##  $ vs  : num  0 0 1 1 0 1 0 1 1 1 ...
##  $ am  : num  1 1 1 0 0 0 0 0 0 0 ...
##  $ gear: num  4 4 4 3 3 3 3 4 4 4 ...
##  $ carb: num  4 4 1 1 2 1 4 2 2 4 ...
```

$Y$(종속변수)는 `mpg`이며 그 외의 변수들로 종속변수를 설명해보자.

```r
fit <- lm(mpg ~ ., data = mtcars)
summary(fit)
```
```r
## 
## Call:
## lm(formula = mpg ~ ., data = mtcars)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -3.4506 -1.6044 -0.1196  1.2193  4.6271 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)  
## (Intercept) 12.30337   18.71788   0.657   0.5181  
## cyl         -0.11144    1.04502  -0.107   0.9161  
## disp         0.01334    0.01786   0.747   0.4635  
## hp          -0.02148    0.02177  -0.987   0.3350  
## drat         0.78711    1.63537   0.481   0.6353  
## wt          -3.71530    1.89441  -1.961   0.0633 .
## qsec         0.82104    0.73084   1.123   0.2739  
## vs           0.31776    2.10451   0.151   0.8814  
## am           2.52023    2.05665   1.225   0.2340  
## gear         0.65541    1.49326   0.439   0.6652  
## carb        -0.19942    0.82875  -0.241   0.8122  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 2.65 on 21 degrees of freedom
## Multiple R-squared:  0.869,  Adjusted R-squared:  0.8066 
## F-statistic: 13.93 on 10 and 21 DF,  p-value: 3.793e-07
```

이며 모델의 성능을 평가할 수 있는 척도 2가지

- `p-value` : 유의수준 0.05하에서 유의한 변수가 없음

- `R-square` : 0.869로 매우 높은 수준

많은 변수들을 사용하지 않고 최소한의 변수로 모델링을 진행하는 것이 효율적이다. 

```r
fullmodel <- lm(mpg~., data = mtcars)
step.models <- step(fullmodel, direction = "backward", trace = F)
summary(step.models)
```
```r
## 
## Call:
## lm(formula = mpg ~ wt + qsec + am, data = mtcars)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -3.4811 -1.5555 -0.7257  1.4110  4.6610 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   9.6178     6.9596   1.382 0.177915    
## wt           -3.9165     0.7112  -5.507 6.95e-06 ***
## qsec          1.2259     0.2887   4.247 0.000216 ***
## am            2.9358     1.4109   2.081 0.046716 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 2.459 on 28 degrees of freedom
## Multiple R-squared:  0.8497, Adjusted R-squared:  0.8336 
## F-statistic: 52.75 on 3 and 28 DF,  p-value: 1.21e-11
```

이며 모델의 성능을 평가할 수 있는 척도 2가지

- `p-value` : 유의수준 0.05하에서 `wt`, `qsec`, `am`이 유의함

- `R-square` : 0.8497로 매우 높은 수준

## 2-2. Classification

$Y$(종속변수)의 형태가 범주형인 데이터에 대해서 실시하는 분석 방법론이다. 

마찬가지로 3단계로 이루어지는데

1. 훈련데이터(train data) 예측데이터(test data)로 분할

2. 훈련데이터(train data)로 모델링

3. 예측데이터(test data)의 성능평가 및 시각화

이며 데이터를 먼저 살펴보자.

```r
rm(list = ls())
#install.packages("nnet")
library(nnet)

data(iris)
pal <- c(2, 3, 4)
plot(iris[,-c(3,4,5)], col = pal[iris[,5]], main = "Multiple Logistic sepal.length vs sepal.width", 
     xlab = "Sepal.Length", ylab = "Sepal.Width")
```

<center><img src="/assets/Application_lecture1/12.png"></center>

```r
fit <- multinom(Species ~., data = iris[,c(1, 2, 5)])
```
```r
## # weights:  12 (6 variable)
## initial  value 164.791843 
## iter  10 value 62.715967
## iter  20 value 59.808291
## iter  30 value 55.445984
## iter  40 value 55.375704
## iter  50 value 55.346472
## iter  60 value 55.301707
## iter  70 value 55.253532
## iter  80 value 55.243230
## iter  90 value 55.230241
## iter 100 value 55.212479
## final  value 55.212479 
## stopped after 100 iterations
```
```r
summary(fit)
```
```r
## Call:
## multinom(formula = Species ~ ., data = iris[, c(1, 2, 5)])
## 
## Coefficients:
##            (Intercept) Sepal.Length Sepal.Width
## versicolor   -92.09924     40.40326   -40.58755
## virginica   -105.10096     42.30094   -40.18799
## 
## Std. Errors:
##            (Intercept) Sepal.Length Sepal.Width
## versicolor    26.27831     9.142717    27.77772
## virginica     26.37025     9.131119    27.78875
## 
## Residual Deviance: 110.425 
## AIC: 122.425
```

범주의 갯수가 3개이므로 2개의 분류 직선이 만들어진다.
$$
\begin{split}
\log\left(\frac{p_{versicolor}}{p_{setosa}}\right) &= -92.09924 + 40.40326Sepal.Length -40.58755Sepal.Width \\
\log\left(\frac{p_{virginica}}{p_{setosa}}\right) &= -105.10096 + 42.300946Sepal.Length -40.18799Sepal.Width
\end{split}
$$

이며 이를 바탕으로 코딩을 실시한다.
```r
rm(list = ls())
library(nnet)
set.seed(1)

n <- nrow(iris) # 행의 갯수
index <- sample(n, 0.7*n, replace = F)
train.data<- iris[index,c(1, 2, 5)]
test.data <- iris[-index,c(1, 2, 5)]
dim(train.data)
```
```r
## [1] 105   3
```
```r
dim(test.data)
```
```r
## [1] 45  3
```
```r
fit <- multinom(Species ~., data = train.data)
```
```r
## # weights:  12 (6 variable)
## initial  value 115.354290 
## iter  10 value 44.002559
## iter  20 value 41.701929
## iter  30 value 37.449424
## iter  40 value 37.292331
## iter  50 value 37.271378
## iter  60 value 37.253756
## iter  70 value 37.236630
## iter  80 value 37.225070
## iter  90 value 37.219598
## iter 100 value 37.212447
## final  value 37.212447 
## stopped after 100 iterations
```
```r
pred<- predict(fit, new = test.data, type = "class")
accuracy <- mean(test.data$Species == pred)
accuracy
```
```r
## [1] 0.8
```
```r
r <- sapply(iris[,1:2], range, na.rm = TRUE)
r
```
```r
##      Sepal.Length Sepal.Width
## [1,]          4.3         2.0
## [2,]          7.9         4.4
```
```r
grid1 <- seq(r[1,1], r[2,1], length.out = 100)
grid2 <- seq(r[1,2], r[2,2], length.out = 100)
grid <- expand.grid(Sepal.Length = grid1, Sepal.Width = grid2)
p <- predict(fit, new = grid, type= "class")
z <- matrix(as.integer(p), nrow = length(grid1), byrow = F)

pal <- c(2, 3, 4)
plot(iris[,-c(3,4,5)], col = pal[iris[,5]], main = "Multiple Logistic", 
     xlab = "Sepal.Length", ylab = "Sepal.Width")
points(grid, col = pal[p], pch = ".")
contour(grid1, grid2, z, add = T, drawlabels = F, lwd = 1)
```

<center><img src="/assets/Application_lecture1/13.png"></center>

# 3. Question

다음의 형식을 이용해 `mtcars`데이터와 `iris`데이터를 이용해 분석 보고서를 완성하시오.
```r
---
title: "Report via example data"
author: "SSWU, Your Name"
date: \today
output: html_document
---
```

```r
```{r, echo = T, out.width = '70%', fig.align='center'}
(your script)
` ````
```

### 1. Regression
#### 1-1. See mtcars data
#### 1-2. Modeling
### 2. Classification
#### 2-1. See iris data
#### 2-2. Modeling
#### 2-3. Plotting
### 3. Conclusion
