---
layout: post
title:  "Basic Lecture8"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. 선형회귀분석

## 1-1. 정의 및 목적

- $X$(독립변수, 원인)와 $Y$(종속변수, 결과) 사이의 관계를 표현한 식

- $Y$(종속변수, 결과)를 $X$(독립변수, 원인)들의 선형 결합으로 표현한 모델
$$
Y = \beta_0 + \beta_1X + \epsilon
$$

- 선형결합이란 비례관계 혹은 반비례관계로 표현하는 형태

- 미래의 값을 예측하는 데에 목적이 있음

- 자세한 이론적인 배경에 대한 부분은 이번 강의에서는 생략

## 1-2. 예시

아래 표의 예시를 살펴보자.

|관측치|집크기(X)|집가격(Y)|
|:--:|:--:|:--:|
|1|1380|76|
|2|3120|216|
|3|3520|238|
|4|1130|69|
|5|1030|50|
|6|1720|119|
|7|3920|282|
|8|1490|81|
|9|1860|132|
|10|3430|228|
|11|2000|145|
|12|3660|251|
|13|2500|170|
|14|1220|71|
|15|1390|79|

위의 표에 대한 그래프는 아래와 같다. 
```r
x <- c(1380, 3210, 3520, 1130, 1030, 1720, 3920, 1490, 1860, 3430, 2000, 3660, 2500, 1220, 1390)
y <- c(76, 216, 238, 69, 50, 119, 282, 81, 132, 228, 145, 251, 170, 71, 29)
plot(x, y, xlab = "Area", ylab = "Value", main = "Scatter plot")
```

<center><img src="/assets/Basic_lecture8/1.png"></center>

데이터의 형태가 집면적이 늘어남에 따라 가격이 올라가는 **선형관계**임을 쉽게 알수 있다.

또한 우리의 관심사는 위의 산점도를 잘 설명할 수 있는 직선을 그리고 싶은 것이다.
```r
fit <- lm(y ~ x)
grid<- seq(min(x), max(x), length.out = 1000)
pred<- predict(fit, new = data.frame(x = grid))
plot(x, y, xlab = "Area", ylab = "Value", main = "Linear regression")
points(grid, pred, type = "l", col = "red", lwd = 2)
```

<center><img src="/assets/Basic_lecture8/2.png"></center>

또한 우리가 갖고 있지 않은 집크기(X)가 2000인 데이터를 예측할 수도 있는 분석이 회귀분석이다.

# 2. R coding

## 2-1. Simple regression

먼저 분석할 데이터를 살펴보자.
```r
data(cars)
str(cars)
```

- $X$(독립변수)는 `speed`이고 $Y$(종속변수)는 `dist`

- 단순선형회귀분석 모델을 만드는 명령어는 `lm()`

- `lm(formula, data)`

    + `formula` : 분석하고 싶은 형태, Y~X 꼴
    
    + `data` : 만약 데이터의 명을 입력하면 $ 기호를 사용할 필요 없이 바로 변수 입력 가능

```r
fit <- lm(cars$dist ~ cars$speed)
fit <- lm(dist ~ speed, data = cars)
summary(fit)
```

<center><img src="/assets/Basic_lecture8/3.PNG"></center>

- p-value는 $Y$를 설명하는데 $X$가 얼마만큼 영향력이 있는지 알려주는 척도

    + Pr(>|t|)의 값이 0.05보다 작으면 영향력이 있다고 판단
    
    + Pr(>|t|)의 값이 0.05보다 크면 영향력이 없다고 판단
    
    + 여기에서는 `speed`의 p-value가 0.05보다 작으므로 영향력이 있다고 판단
    
- R-squred는 만든 모델이 얼마만큼 좋으며 설명력이 어느정도 인지 알려주는 척도

    + 기준은 없지만 보통 0.6이상이면 좋다고 판단
    
    + 여기에서는 0.6511로 만든 모델이 설명력이 좋다고 판단

- 예측 회귀선을 그리기 위해 $grid$와 `predict()`함수를 사용

- `predict(model, new)`

    + `model` : 회귀분석을 통해 만들어진 모델
    
    + `new` : 예측하고자 하는 독립변수의 형태
    
        + 데이터 프레임 형태로 작성되어야 함
        
        + 모델에 사용한 독립변수의 이름을 같게 만들어줘야함
        
```r
grid <- seq(min(cars$speed), max(cars$speed), length.out = 100)
pred <- predict(fit, new = c(grid))
```

<center><img src="/assets/Basic_lecture8/4.PNG"></center>

위의 에러는 예측하고자 하는 데이터의 형태가 **데이터프레임**이 아니기 때문에 발생하는 에러이다. 
```r
pred <- predict(fit, new = data.frame(grid))
colnames(data.frame(grid))
```

<center><img src="/assets/Basic_lecture8/5.PNG"></center>

위의 에러는 예측하고자 하는 데이터의 형태가 데이터프레임이지만 독립변수의 이름이 모델에서 **사용한 독립변수의 이름**과 다르기 때문에 발생하는 에러이다.
```{r, error = T}
pred <- predict(fit, new = data.frame(speed = grid))
colnames(data.frame(grid))
```

위의 예측한 데이터를 바탕으로 그래프를 그려보자.
```{r, out.width = "70%", fig.align='center'}
plot(cars$speed, cars$dist, xlab = "Speed", ylab = "Dist", main = "Linear regression for car data")
points(grid, pred, type = "l", col = "red", lwd = 2)
```

<center><img src="/assets/Basic_lecture8/6.png"></center>

## 2-2. Multiple regression

1차원의 독립변수 데이터가 아니라, 다차원의 독립변수 데이터를 살펴보자.

```r
data(mtcars)
is.data.frame(mtcars)
str(mtcars)
```

마찬가지로 위의 식을 사용해본다.

```r
fit <- lm(mpg ~ cyl + disp + hp + drat + wt + qsec + vs + am + gear + carb, data = mtcars)
```

일일히 변수를 쓰는 것은 좋지 못하다. 따라서 아래의 방식을 사용한다. 
```r
fit <- lm(mpg ~ ., data = mtcars)
summary(fit)
```

<center><img src="/assets/Basic_lecture8/7.PNG"></center>

모델의 적합성을 나타내는 R-squared는 0.869로 매우 높게 나오지만, 변수들의 중요도를 나타내는 p-value는 `wt`변수를 제외하고 좋지 못하다.

Occam's Razor(오컴의 면도날)은 통계학의 모델링에서도 적용되는 말이다.

모든 변수를 사용하면 좋겠지만 해석력이 떨어지고 computation cost도 증가하게 된다.

따라서 적은 변수들로 종속변수를 많이 설명할 수 있는 모델을 만드는 것이 회귀분석의 목적이다.

변수 제거 방법에는 크게 3가지가 존재한다.

- Forward selection(전진 선택법)

- Backward elimination(후진 제거법)

- Stepwise selection(단계적 선택법)

이에 대한 자세한 내용은 생략하고 후진 제거법에 대해서만 실시한다.

제일 p-value가 높은 변수를 제거하고 재적합시킨다.

```r
adj.data1 <- subset(mtcars, select = -cyl)
str(adj.data1)
adj.fit1 <- lm(mpg~., data = adj.data1)
summary(adj.fit1)
```
<center><img src="/assets/Basic_lecture8/8.PNG"></center>

```r
adj.data2 <- subset(mtcars, select = -c(cyl, vs))
str(adj.data2)
adj.fit2 <- lm(mpg~., data = adj.data2)
summary(adj.fit2)
```
<center><img src="/assets/Basic_lecture8/9.PNG"></center>

이러한 과정을 계속해서 반복하면 된다. 일일히 하는 것은 불가능하므로 `step`함수를 이용한다.

```r
fullmodel <- lm(mpg~., data = mtcars)
step.models <- step(fullmodel, direction = "backward", trace = F)
summary(step.models)
```
<center><img src="/assets/Basic_lecture8/10.PNG"></center>

3개 변수가 모두 유의하며 R-squared의 값도 0.8497로 매우 높으므로 전체 변수를 사용하지 말고 3개의 변수만을 사용한다.

# 3. 로지스틱 회귀분석

## 3-1. 정의 및 목적

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

## 3-2. 예시

아래 표의 예시를 살펴보자.

|나이|질병유무|
|:--:|:--:|
|22|0|
|24|0|
|27|0|
|28|0|
|32|0|
|33|1|
|38|0|
|46|1|
|47|0|
|51|0|
|58|1|
|60|1|
|67|1|
|77|1|

위의 표에 대한 그래프는 아래와 같다.
```r
x <- c(22, 24, 27, 28, 32, 33, 38, 46, 47, 51, 58, 60, 67, 77)
y <- c(0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1)
plot(x, y, xlab = "Age", ylab = "disease", main = "Scatter plot")
grid <- seq(min(x), max(x), length.out = 100)
fit <- lm(y ~ x)
pred<- predict(fit, new = data.frame(x = grid))
points(grid, pred, lwd = 2, col = "red", type = "l")
```
<center><img src="/assets/Basic_lecture8/11.png"></center>

위의 그래프를 설명하는 직선식은 너무 설명력이 떨어진다. 

그 이유는 앞으로 등장하게 될 새로운 값이 내가 갖고 있는 범위를 벗어날 때 문제가 생기기 때문이다.
```r
plot(x, y, xlab = "Age", ylab = "disease", main = "Wrong plot",
     xlim = c(min(x) - 6, max(x) + 6), ylim = c(min(y) - 0.5, max(y) + 0.5))
grid <- seq(min(x) - 6, max(x) + 6, length.out = 100)
fit <- lm(y ~ x)
pred<- predict(fit, new = data.frame(x = grid))
points(grid, pred, lwd = 2, col = "red", type = "l")
```
<center><img src="/assets/Basic_lecture8/12.png"></center>

0과 1사이의 확률을 예측하고자 하는데 확률의 범위를 넘어선 예측값을 반환해준다. 

이에 대한 설명력을 높여주며 어떠한 값이 등장하더라도 0과 1사이의 확률로 반환해주는 모델링이 로지스틱 회귀모델링이다.
```r
plot(x, y, xlab = "Age", ylab = "disease", main = "Logistic regression")
grid <- seq(min(x), max(x), length.out = 100)
fit <- glm(y ~ x, family = binomial)
pred<- predict(fit, new = data.frame(x = grid), type = "response")
points(grid, pred, lwd = 2, col = "red", type = "l")
```
<center><img src="/assets/Basic_lecture8/13.png"></center>

종속변수가 1을 가질 확률로 모델을 만들었으므로 확률이 높으면 1을 줄 수 있고 확률이 작으면 0으로 줄 수 있으므로 설명력이 생겼다.

# 4. R-code

## 4-1. Explore data

먼저 분석할 데이터를 살펴보자. `iris`데이터를 사용하되 범주를 2개로만 제한하기 위해서 `Species`변수가 `virginica`와 같으면 1, 다르면 0으로 만들어준다.
```r
rm(list = ls())
data(iris)
iris$Species <- ifelse(iris$Species=="virginica", 1, 0)
iris$Species <- factor(iris$Species)
pal <- c("red", "blue")
plot(iris[,1:2], col = pal[iris[,5]], main = "Iris data")
legend("topleft", legend = c("virginica", "no virginica"), cex = 0.8, col = pal, lwd = 2)
segments(x0 = 5.5, y0 = 2, x1 = 6.5, y1 = 5, col = "green", lty = 2, lwd = 2)
```
<center><img src="/assets/Basic_lecture8/14.png"></center>

위의 분류를 가능하게 하는 녹색 직선을 찾는 모델이 로지스틱 회귀분석모델이다.

## 4-2.  Split train data and test data

- 머신러닝의 기본은 모델의 성능을 평가하는 것

- Accuracy : 평가의 척도
$$
Accuracy = \frac{1}{n}\sum_{i = 1}^nI(y_i == \hat{y}_i)
$$

    + 예측한 범주와 원래 범주가 같은 갯수의 평균
    
- train data와 test data로 분류

    + 보통 데이터의 비율이 7:3이 되게 분류
    
    + 학습 데이터(train data)가 전체를 잘 대변할 수 있는 비율이면 됨 

```r
set.seed(1)
n <- nrow(iris) # 행의 갯수
index <- sample(n, 0.7*n, replace = F)
train.data<- iris[index,]
test.data <- iris[-index,]
dim(train.data)
dim(test.data)
```

## 4-3. Modeling

- `glm(formula, data, family = binomial)`

    + `family` : 지수족에 대한 부분, 로지스틱은 $binomial$
    
- `predict(model, new, type = "response")`

    + `type` : glm에서 반환하는 값은 $p/(1-p)$이며 확률로 반환하기 위해 `response`로 지정 
    
```r
fit <- glm(Species ~ Sepal.Length + Sepal.Width, data = train.data, family = binomial)
pred<- predict(fit, new = test.data, type = "response")
pred
```

변수 `pred`의 형태는 0부터 1사이의 확률이다. 

이를 범주화 시키기 위해 0.5이상을 1, 0.5미만을 0으로 잡는다.
```r
y.hat <- ifelse(pred>=0.5, 1, 0)
accuracy <- mean(test.data$Species == y.hat)
accuracy
```

정확도는 0.75로 예측데이터(test data)의 75%를 올바르게 분류해냈다.

또한 이러한 모델로 만들어진 분류 직선을 그려보자.
```r
grid1 <- seq(min(iris$Sepal.Length)-1, max(iris$Sepal.Length)+1, length.out = 200)
grid2 <- seq(min(iris$Sepal.Width)-1, max(iris$Sepal.Width)+1, length.out = 200)
grid <- expand.grid(Sepal.Length = grid1, Sepal.Width = grid2)
p <- ifelse(predict(fit, new = grid, type= "response")>=0.5, 1, 0)
p <- factor(p)
z <- matrix(as.integer(p), nrow = length(grid1), byrow = F)

pal <- c("red", "blue")
plot(iris[,-c(3,4,5)], col = pal[iris[,5]], main = "Logistic sepal.length vs sepal.width")
points(grid, col = pal[p], pch = ".")
contour(grid1, grid2, z, add = T, drawlabels = F, lwd = 1)
```
<center><img src="/assets/Basic_lecture8/15.png"></center>

# 5. 다항 로지스틱 회귀분석

## 5-1. 정의 및 설명

범주의 수가 위와 같은 2개가 아니라, 3개 이상일 때 사용하는 방법론이다.

각 범주들의 확률값을 계산한다

$$
p_1, p_2, \ldots, p_k, \quad \sum_{j = 1}^kp_j = 1
$$

그 후, 로지스틱 회귀분석에서 사용했던 회귀식을 `k-1`개 만큼 생성한다.

$$
\begin{split}
\log\left(\frac{p_1}{p_k}\right) &= \beta_{01} + \cdots + \beta_{p1} x_p \\
\log\left(\frac{p_2}{p_k}\right) &= \beta_{02} + \cdots + \beta_{p2} x_p \\
&\vdots \\
\log\left(\frac{p_{k-1}}{p_k}\right) &= \beta_{0k-1} + \cdots + \beta_{pk-1} x_p \\
\end{split}
$$

위의 식을 이용해 전체 데이터를 분류할 수 있는 초평면(Hyper plane)을 `k-1`개를 생성해 `k`개의 범주를 분류한다.

## 5-2. R-code

`library(nnet)`을 사용한다.

- `multinom(formula, data, weights, ...)`

    + `weights` : 가중치들의 갯수를 결정하는 부분

```r
rm(list = ls())
#install.packages("nnet")
library(nnet)

data(iris)
pal <- c(2, 3, 4)
plot(iris[,-c(3,4,5)], col = pal[iris[,5]], main = "Multiple Logistic sepal.length vs sepal.width", 
     xlab = "Sepal.Length", ylab = "Sepal.Width")
fit <- multinom(Species ~., data = iris[,c(1, 2, 5)])
summary(fit)
```
<center><img src="/assets/Basic_lecture8/15.png"></center>
<center><img src="/assets/Basic_lecture8/16.PNG"></center>

위에서 언급한 대로 회귀식이 2개가 생긴다.

$$
\begin{split}
\log\left(\frac{p_{versicolor}}{p_{setosa}}\right) &= -92.09924 + 40.40326Sepal.Length -40.58755Sepal.Width \\
\log\left(\frac{p_{virginica}}{p_{setosa}}\right) &= -105.10096 + 42.300946Sepal.Length -40.18799Sepal.Width
\end{split}
$$

각각의 p-value를 알아보도록 하자.

```r
summ <- summary(fit)
pt(abs(summ$coefficients / summ$standard.errors), df=1, lower=FALSE)
```

<center><img src="/assets/Basic_lecture8/17.PNG"></center>

그다지 유의하지 않음을 알 수 있으며 이를 바탕으로 정확도와 분류 그래프를 그려보자.

```{r, out.width = "70%", fig.align='center', warning=F}
rm(list = ls())
library(nnet)
set.seed(1)

n <- nrow(iris) # 행의 갯수
index <- sample(n, 0.7*n, replace = F)
train.data<- iris[index,c(1, 2, 5)]
test.data <- iris[-index,c(1, 2, 5)]
dim(train.data)
dim(test.data)

fit <- multinom(Species ~., data = train.data)
pred<- predict(fit, new = test.data, type = "class")
pred
accuracy <- mean(test.data$Species == pred)
accuracy

r <- sapply(iris[,1:2], range, na.rm = TRUE)
r

grid1 <- seq(r[1,1], r[2,1], length.out = 200)
grid2 <- seq(r[1,2], r[2,2], length.out = 200)
grid <- expand.grid(Sepal.Length = grid1, Sepal.Width = grid2)
p <- predict(fit, new = grid, type= "class")
z <- matrix(as.integer(p), nrow = length(grid1), byrow = F)

pal <- c(2, 3, 4)
plot(iris[,-c(3,4,5)], col = pal[iris[,5]], main = "Multiple Logistic sepal.length vs sepal.width", 
     xlab = "Sepal.Length", ylab = "Sepal.Width")
points(grid, col = pal[p], pch = ".")
contour(grid1, grid2, z, add = T, drawlabels = F, lwd = 1)
```

<center><img src="/assets/Basic_lecture8/18.png"></center>
