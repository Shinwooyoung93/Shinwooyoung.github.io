---
layout: post
title:  "Basic Lecture5"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. 함수의 생성

## 1-1. 함수의 목적

1. 작업을 작은 단위로 분할하여 수행해 효율성을 제고

2. 자유로운 수정 $\cdot$ 보완이 가능하므로 작업의 유연성 확보

3. 작업에 사용하는 코드와 오류 등의 발생 원인 파악이 가능

4. 코드의 크기를 줄임으로써 작업 프로세스 이해가 용이

5. 사용자가 정의한 함수를 자유롭게 활용해 작업목적을 충족

```r
x1 <- 2:20
mean(x1)
```
```r
## [1] 11
```
```r
max(x1)
```
```r
## [1] 20
```
```r
min(x1)
```
```r
## [1] 2
```
```r
sum(x1)
```
```r
## [1] 209
```
```r
x2 <- 20:50
mean(x2)
```
```r
## [1] 35
```
```r
max(x2)
```
```r

```
```r
min(x2)
```
```r
## [1] 50
```
```r
sum(x2)
```
```r
sum(x2)
```
```r

## [1] 1085
```
```r
x3 <- 3:5
mean(x3)
```
```r
## [1] 4
```
```r
max(x3)
```
```r
## [1] 5
```
```r
min(x3)
```
```r
## [1] 3
```
```r
sum(x3)
```
```r
## [1] 12
```

이러한 코딩을 해도 괜찮지만 계산하고자 하는 변수의 수가 많아질 경우 이를 읽기도 힘들고 스크립트의 크기도 많이 차지해 효율적이지 못하다.
```r
my.function <- function(x){
  list(mean = mean(x), max = max(x), min = min(x), sum = sum(x))
}
x1 <- 2:20
x2 <- 20:50
x3 <- 3:5
my.function(x1)
```
```r
## $mean
## [1] 11
## 
## $max
## [1] 20
## 
## $min
## [1] 2
## 
## $sum
## [1] 209
```
```r
my.function(x2)
```
```r
## $mean
## [1] 35
## 
## $max
## [1] 50
## 
## $min
## [1] 20
## 
## $sum
## [1] 1085
```
```r
my.function(x3)
```
```r
## $mean
## [1] 4
## 
## $max
## [1] 5
## 
## $min
## [1] 3
## 
## $sum
## [1] 12
```

## 1-2. 사용자 정의 함수 만들기

$$
\begin{split}
\text{함수이름}& \leftarrow function(\text{함수 인수})\{\\
  &\text{함수 몸체} \\
  &return(\text{출력값}) \\
  &\}
\end{split}
$$

간단한 예시를 반복하며 살펴보자. 데이터를 입력하면 1을 추가해서 반환하는 함수를 만들어보자.
```r
rm(list = ls())
f <- function(x){
  x + 1
}
x
```
```r
## Error in eval(expr, envir, enclos): 객체 'x'를 찾을 수 없습니다
```
```r
f(c(10))
```
```r
## [1] 11
```
```r
f(c(1, 2, 3))
```
```r
## [1] 2 3 4
```

함수 인자로 사용된 변수 `x`는 함수 안에서만 정의되지, 작업공간에서 정의되지 않는다. 따라서 `x`는 찾을 수 없다.

또한 `return`의 역할은 함수에서 출력할 것을 명령하는 함수이다. 만약 `return`이 없다면 함수의 가장 마지막 줄을 반환하게 된다.
```r
quad.function <- function(x){
  print("이차함수 y = x^2")
  y <- x^2
  return(y)
}
quad.function(10)
```
```r
## [1] "이차함수 y = x^2"
## [1] 100
```

또한 `return`을 입력하게 되면 `return` 뒤의 내용은 실행되지 않는다.
```r
quad.function <- function(x){
  y <- x^2
  return(y)
  print("이차함수 y = x^2")
}
quad.function(10)
```
```r
## [1] 100
```

맨 위의 예시처럼 반환하고자 하는 값이 2개 이상일 경우가 있다. 만약 `return`하나로만 하게 되면 작동하지 않는다.
```r
my.function <- function(x){
  return(mean = mean(x), max = max(x), min = min(x), sum = sum(x))
}
x1 <- 2:20
my.function(x1)
```
```r
## Error in return(mean = mean(x), max = max(x), min = min(x), sum = sum(x)): 다중인자 반환은 허용되지 않습니다
```

따라서 이를 해결해주기 위해 `list()`데이터 구조를 이용한다. `c()`를 이용해 묶어도 되지만 사용하고자 하는 값을 제어할 때 유용하다.
```r
my.function <- function(x){
  return(c(mean = mean(x), max = max(x), min = min(x), sum = sum(x)))
}
x1 <- 2:20
y <- my.function(x1)
y
```
```r
## mean  max  min  sum 
##   11   20    2  209
```

우리의 관심사가 만약 minimum에만 있다면 위의 코드를 사용할 수 없다. 
```r
my.function <- function(x){
  return(list(mean = mean(x), max = max(x), min = min(x), sum = sum(x)))
}
x1 <- 2:20
y <- my.function(x1)
y$min
```
```r
## [1] 2
```

조금 더 확장해서 거리를 알려주는 함수를 만들어보자. 거리 함수는 다음의 공식을 사용한다.
$$
d_p(x, y) = \sqrt[p]{\sum_i^n(x_i - y_i)^p}
$$

이고 $p = 1$이면 $|\sum_i(x_i - y_i)|$. $p = 2$이면 유클리디언거리 $\sqrt{\sum_i(x_i - y_i)^2}$이다. 여기에서 사용되는 변수는 총 3개이다. `x, y, p`
```r
dist.function <- function(x, y, p = 2){
  dist <- (sum((x - y)^p))^{1/p}
  return(dist)
}
x <- c(1, 2, 3, 4)
y <- c(2, 3, 4, 5)
dist.function(x = x, y = y)
```
```r
## [1] 2
```
```r
dist.function(x = x, y = y, p = 2)
```
```r
## [1] 2
```
```r
dist.function(x = x, y = y, p = 1)
```
```r
## [1] -4
```

함수 인자에 사용된 $p = 2$를 자세히보자. 이는 기본 값을 지정하는 것으로 만약 `p`를 입력하지 않아도 기본 값이 2임을 의미한다.

끝으로 상관계수를 계산해주는 함수를 만들고 마치겠다. 상관계수 공식은 다음과 같다.
$$
cor(x, y) = \frac{\sum_i(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i(x_i - \bar{x})^2(y_i - \bar{y})^2}}
$$

```r
cor.function <- function(x, y){
  cxy <- sum((x - mean(x))*(y - mean(y)))
  vx <- sum((x - mean(x))^2)
  vy <- sum((y - mean(y))^2)
  crr <- cxy/sqrt(vx*vy)
  return(crr)
}
```

# 2. Question

벡터를 입력하였을 때 최솟값과 최댓값 그리고 `k`번째 최댓값을 반환하는 `my.function`함수를 만드시오. 단, `max(), min()`사용하지 말고 `sort()`만을 사용하되 `k`의 기본값은 2로 설정하시오.
```r
x <- c(14,5,2,1,34,6,2,0,3)
```
```r
## $minimum
## [1] 0
## 
## $maximum
## [1] 34
## 
## $`k-th maximum`
## [1] 14
```
