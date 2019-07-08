---
layout: post
title:  "Basic Lecture4"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. 조건문과 반복문

## 1-1. 조건문

- 특정한 조건을 만족했을 경우에만 프로그램 코드를 수행하는 제어 구문

- 항상 논리 연산이 수반됨

- 가장 많이 사용하는 프로그래밍 문장

- `if`(*조건*) **명령문**

- `if`(*조건*) **명령문 1** `else` **명령문 2**

- `if`(*조건*) **명령문 1** `else if` **명령문 2** ($\cdots$) `else if` **명령문 N-1** `else` **명령문 N**

- `ifelse`(*조건*, **명령문1**, **명령문2**)

    + 조건이 참일 경우 명령문 1을 수행, 거짓이면 명령문 2를 수행

### 1-1-1. If문

간단한 예시를 통해 살펴보자.
```r
x <- c(1, 2, 3, 4)
y <- c(2, 1, 4, 4)
if(sum(x) < sum(y)) print(x)
```
```r
## [1] 1 2 3 4
```
```r
if(sum(x) > sum(y)) print(x)
if(sum(x) > sum(y)) print("sum(x) is bigger than sum(y)") else print("sum(y) is bigger or equal than sum(x)")
```
```r
## [1] "sum(y) is bigger or equal than sum(x)"
```

조금 더 확장된 예시를 살펴보자. 점수에 따라 학점을 주는 코드이다.
```r
score <- 82
if(score >= 90){
  grade = "A"
}else if(score >= 80){
  grade = "B"
}else{
  grade = "C"
}
grade
```
```r
## [1] "B"
```

주의할 부분은 조건문을 이어나갈 때 `}else if`와 같은 형식으로 진행해야한다.

### 1-1-2. Ifelse문

위의 조건문을 간략하게 만드는 구문이 `ifelse()`함수이다.
```r
x <- 80
ifelse(x>=90, "equal or bigger than 90", "less than 90")
```
```r
## [1] "less than 90"
```

점수에 따라 학점을 주는 코드를 `ifelse()`로 바꾸면
```r
score <- 82
grade <- ifelse(score>=90, "A", ifelse(score>=80, "B", "C"))
grade
```
```r
## [1] "B"
```

`ifelse()`구문은 벡터를 다루는데 좋다. 예시를 살펴보자.
```r
score <- c(75, 82, 92, 66)
if(score >= 90){
  grade <- "A"
}else if(score >= 80){
  grade <- "B"
}else{grade <- "C"}
```
```r
## Warning in if (score >= 90) {: length > 1 이라는 조건이 있고, 첫번째 요소만
## 이 사용될 것입니다
## Warning in if (score >= 80) {: length > 1 이라는 조건이 있고, 첫번째 요소만
## 이 사용될 것입니다
```
```r
grade
```
```r
## [1] "C"
```

이처럼 `if`문만을 사용할 경우, 여러값이 있는 벡터를 다루지 못한다. 

이를 해결하기 위해선 요소 하나 하나 마다 조건문을 입력해주어야하지만 아래의 `ifelse()`함수는 그럴 필요가 없다.
```r
grade <- ifelse(score >= 90, "A", ifelse(score >= 80, "B", "C"))
grade
```
```r
## [1] "C" "B" "A" "C"
```

## 1-2. 반복문

### 1-2-1. While문

- `while()`함수는 조건이 성립되는 동안만 식을 수행한다.

- `break`문을 이용하여 조건을 제한할 수 있다.

- `while(condition){body}`

- `while(조건문){실행문}`

```r
x <- 1
while(x <= 5){
  print(x)
  x <- x + 1
}
```
```r
## [1] 1
## [1] 2
## [1] 3
## [1] 4
## [1] 5
```

```r
x <- 1
while(TRUE){
  print(x)
  x <- x + 1
  if(x == 6) break
}
```
```r
## [1] 1
## [1] 2
## [1] 3
## [1] 4
## [1] 5
```

### 1-2-2. For문

- `for` (*variable* `in` *vector*) `{body}`

- `for(변수 in 반복횟수){실행문}`

```r
for(i in 1:5){
  print(i)
}
```
```r
## [1] 1
## [1] 2
## [1] 3
## [1] 4
## [1] 5
```

```r
for(i in 1:10){
  print(i)
  if(i == 6) break
}
```
```r
## [1] 1
## [1] 2
## [1] 3
## [1] 4
## [1] 5
## [1] 6
```

# 2. Question

## 2-1. Question

구구단 7단을 만드시오
```r
## [1] "7 * 1 = 7"
## [1] "7 * 2 = 14"
## [1] "7 * 3 = 21"
## [1] "7 * 4 = 28"
## [1] "7 * 5 = 35"
## [1] "7 * 6 = 42"
## [1] "7 * 7 = 49"
## [1] "7 * 8 = 56"
## [1] "7 * 9 = 63"
```

## 2-2. Question

짝수 짝수 7단을 만드시오
```r
## [1] "7 * 2 = 14"
## [1] "7 * 4 = 28"
## [1] "7 * 6 = 42"
## [1] "7 * 8 = 56"
```
