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

`---` <br/>
`title: "Untitled"` <br/>
`author: "KU, Shin wooyoung"` <br/>
`date: "2019년 6월 2일"` <br/>
`output: html_document` <br/>
`---`

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

`안녕하세요`<br/>
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

` ```{r} ` <br/>
` rm(list = ls())` <br/>
` data(cars)` <br/>
` summary(cars)` <br/>
` ``` ` 

* `r` : 내가 입력할 언어가 `R`이라는 것을 명시해주는 부분

```r
rm(list = ls()) 
data(cars) 
summary(cars) 
```

그러나 나의 코드를 보여주고 싶지 않을 상황도 존재한다. 그럴 경우에는 다음의 코드를 사용한다.

` ```{r, echo = F} ` <br/>
` rm(list = ls())` <br/>
` data(cars)` <br/>
` summary(cars)` <br/>
` ``` ` 

* `echo` : 코드의 출력을 담당하는 부분

```r
rm(list = ls()) 
data(cars) 
summary(cars) 
```

### 1-2-4. Input graph

그래프를 입력하거나, 외부 사진을 넣는 방법은 아래의 코드를 활용한다.

` ```{r, out.width = '70%', fig.align='center'} ` <br/>
` knitr::include_graphics("iris.png")` <br/>
` ``` ` 

* `out.width` : 출력물의 크기를 조정하는 부분

* `fig.align` : 출력물의 위치를 조정하는 부분

주의해야할 점은 사진의 위치가 내가 작업하는 공간과 같은 공간에 있어야 한다.

<center><img src="/assets/Application_lecture1/iris.png"></center>

또한 `R-code`를 입력하여 그래프를 그릴 수도 있다.

` ```{r, echo=FALSE}` <br/>
` plot(pressure)` <br/>
` ``` ` <br/>

```{r, echo=FALSE, out.width='50%', fig.align='left'}
data(pressure)
plot(pressure, main = "pressure data plot") 
```

# 2. Exercise data

분석보고서를 만드는 데 필요한 `mtcars`데이터와 `iris`데이터를 이용해 회귀분석 및 로지스틱 회귀분석을 복습해보자.

## 2-1. Regression

회귀분석은 3단계로 이루어지며 

1. 원인과 결과의 설정

2. trend를 잘 설명할 수 있는 모델링

3. 만든 모델로 예측

이며 데이터의 형태를 먼저 살펴보자.

```{r}
data(mtcars)
str(mtcars)
```

$Y$(종속변수)는 `mpg`이며 그 외의 변수들로 종속변수를 설명해보자.

```{r}
fit <- lm(mpg ~ ., data = mtcars)
summary(fit)
```

이며 모델의 성능을 평가할 수 있는 척도 2가지

- `p-value` : 유의수준 0.05하에서 유의한 변수가 없음

- `R-square` : 0.869로 매우 높은 수준

많은 변수들을 사용하지 않고 최소한의 변수로 모델링을 진행하는 것이 효율적이다. 

```{r, out.width="70%", fig.align='center'}
fullmodel <- lm(mpg~., data = mtcars)
step.models <- step(fullmodel, direction = "backward", trace = F)
summary(step.models)
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

```{r, out.width = "70%", fig.align='center', warning=F}
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

범주의 갯수가 3개이므로 2개의 분류 직선이 만들어진다.
$$
\begin{split}
\log\left(\frac{p_{versicolor}}{p_{setosa}}\right) &= -92.09924 + 40.40326Sepal.Length -40.58755Sepal.Width \\
\log\left(\frac{p_{virginica}}{p_{setosa}}\right) &= -105.10096 + 42.300946Sepal.Length -40.18799Sepal.Width
\end{split}
$$

이며 이를 바탕으로 코딩을 실시한다.
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
accuracy <- mean(test.data$Species == pred)
accuracy

r <- sapply(iris[,1:2], range, na.rm = TRUE)
r

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

# 3. Question

다음의 형식을 이용해 `mtcars`데이터와 `iris`데이터를 이용해 분석 보고서를 완성하시오.

`---` <br/>
`title: "Report via example data"` <br/>
`author: "SSWU, Your Name"` <br/>
`date: \today` <br/>
`output: html_document` <br/>
`---`

` ```{r, echo = T, out.width = '70%', fig.align='center'} ` <br/>
` (your script)` <br/>
` ``` ` 

### 1. Regression
#### 1-1. See mtcars data
#### 1-2. Modeling
### 2. Classification
#### 2-1. See iris data
#### 2-2. Modeling
#### 2-3. Plotting
### 3. Conclusion
