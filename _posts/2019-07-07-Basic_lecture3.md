---
layout: post
title:  "Basic Lecture3"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

---
title: "Lecture3 - 데이터 구조"
author: "KU, Shin wooyoung"
date: "2019-07"
output:
  html_document:
    latex_engine: xelatex
mainfont: NanumGothic
---

# 1. 데이터 구조

## 1-1. R과 오브젝트

- R은 오브젝트(Object)중심의 언어

- 오브젝트의 형식은 숫자형(Numeric), 문자형(Characteristic), 요인형(Factor), 논리형(Logical), 실수형(Double), 복소수형(Complex)등이 존재

- 오브젝트에 저장될 수 있는 내용은 벡터(Vector), 행렬(Matrix), 데이터(Data), 배열(Array), 함수(Function) 등이 있으며 이를 오브젝트 할당(Assignment)이라 칭함

- 오브젝트의 이름을 설정할 때 R에서 쓰고 있는 형식은 피하는 것을 권장(list, mean, sum)

## 1-2. 자료 객체(Data object)

|자료 객체|구성 차원|자료 유형|여러 유형 호환 여부|
|:--:|:--:|:--:|:--:|
|벡터(Vector)|1차원|수치/문자/복소수/논리|불가능|
|행렬(Matrix)|2차원|수치/문자/복소수/논리|불가능|
|데이터프레임(Dataframe)|2차원|수치/문자/복소수/논리|**가능**|
|배열(Array)|2차원 이상|수치/문자/복소수/논리|불가능|
|요인(Factor)|1차원|수치/문자|불가능|
|시계열(Time series)|2차원|수치/문자/복소수/논리|불가능|
|리스트(List)|2차원 이상|수치/문자/복소수/논리/함수/표현식/기타|**가능**|

1. 벡터(Vector)

    + 동일한 자료형의 단일 값들이 한군데 모여있는 1차원 자료구조
    + R의 가장 기본적인 자료 저장 방법
    + 일반적으로 하나의 속성을 저장하는 단위 

2. 행렬(Matrix)

    + 행과 열의 2차원 자료 구조
    + R이 데이터분석에서 가장 큰 장점을 갖는 부분
    
3. 데이터프레임(Dataframe)
    
    + 변수와 관찰치로 구성된 2차원의 자료 객체
    + 서로 다른 벡터로 구성된 자료들을 열로 배치한 자료 구조
    + 자료 분석에 있어 가장 일반적인 데이터 구조
    + 후에 배우게 될 ggplot에서 기본이 되는 구조
    
4. 배열(Array)

    + 일반적으로 3차원 이상의 자료 구조

5. 요인(Factor)

    + 벡터 객체 중 범주형 데이터를 원소로 갖는 자료 객체
    + 순서형과 명목형으로 구분
    
6. 시계열(Time series)

    + 일, 월, 년, 시간 등과 같이 일련의 시계열 자료를 표현하는 자료 구조

7. 리스트(List)

    + 서로 다른 데이터의 유형의 결합으로 구성이 가능
    + 자료객체 중 가장 자유로운 구조
    + 후에 배우게 될 함수의 생성에서 필수적인 요소
    
<center><img src="/assets/Basic_lecture3/1.png" width="500" height="500"></center>

자료 객체의 형식을 변환하는 함수는 다음과 같다.

|자료 객체|함수|
|:--------------:|:----------------------------:|
|벡터(Vector)|as.vector(object),as.numeric(object), as.charactor(object),as.logical(object), as.complex(object),as.integer(object)|
|행렬(Matrix)|as.matrix(data object), as.data.matrix(data object)|
|데이터프레임 (Dataframe)|as.data.frame(object)|
|배열(Array)|as.array(object)|
|리스트(List)|as.list(object)|
|요인(Factor)|as.factor(object), as.ordered(object)|

# 2. R coding

## 2-1. Data type

예시를 통해 살펴보자. `x`에 숫자 10 이하의 자연수들을 벡터로 할당하는데에는 3가지 방법이 존재한다. 
```r
x1 <- 1:10
x2 <- seq(from = 1, to = 10, by = 1) # equal to seq(1,10)
x3 <- c(1,2,3,4,5,6,7,8,9,10)
```

벡터의 형태인지 알아보자.
```r
is.vector(x1)
```

이 수들은 어떤 형태인지 알아보자.
```r
mode(x1)
is.factor(x1)
```

오브젝트의 형태를 변경해보고 확인해보자.
```r
y <- as.character(x1)
y
is.character(y)
```

## 2-2. NA, Inf, NaN

데이터의 형태가 위에 언급한 것에 없을 수 있다. 로그분포를 생각해보자.

<center><img src="/assets/Basic_lecture3/2.png" width="500" height="500"></center>

로그분포의 정의역은 0보다 큰값으로 정의된다. 만약 $\log(0)$ 혹은 $\log(-1)$을 계산해보자.
```r
log(0)
log(-1)
```

이처럼 R에서 무한대의 값은 Inf로, 숫자가 정의되지 않을 때는 NaN으로 표시된다. 추가적으로 구조는 존재하지만 값이 없는 값 즉, 결측치는 NA로 정의된다.
```r
x <- c(1, 2, 3, NA)
x
is.na(x)
```

## 2-3. Vector

벡터를 이용한 자료 연산을 진행해보자.
$$
BMI = \frac{체중(kg)}{신장(m)^2}
$$

```r
Height <- c(168, 173, 160, 145, 180)
Weight <- c(80, 65, 92, 53, 76)
BMI <- Weight/(Height/100)^2
BMI
```

이를 데이터프레임 형태로 나타내보자.
```r
data <- data.frame(Height, Weight, BMI)
data
```

`data`에서 `BMI`변수를 추출하고 싶으면 `$`를 활용한 다음의 코드를 사용한다.
```r
data$BMI
```

여러가지 함수를 이용한 벡터의 연산을 진행해보자.
```r
x <- c(1,2,3,10,4)
sum(x)
prod(x)
cumsum(x)
cumprod(x)
exp(x)
sort(x)
paste("No", x)
```

`paste()`함수는 글자를 붙이는 함수이다.

데이터분석에 잘 활용되는 함수인 `rep()`에 대해 자세히 알아보자.

벡터 생성 함수 `rep()`
- 기존에 있는 벡터를 반복하여 새로운 벡터 생성

- `rep(x, times = , length = , each = )`

    + `x`: 반복할 벡터 개체 
    + `times` : 전달된 벡터 `x`의 전체 반복 횟수
    + `each` : 전달된 벡터 `x`의 개별 원소들의 반복 횟수
    + `length` : 출력자료 객체의 길이(크기, 갯수)

```r
x <- 1:3
rep(x, times = 3)
rep(x, times = 3, each = 2)
rep(x, each = 2)
rep(x, each = 2, length = 5)
```

다음은 `seq`함수를 이용한 벡터 생성방법이다.

벡터 생성 함수 `seq()`

- `seq(from = , to = , by = , length.out = , along.with=,)`

    + `from` : 시작할 숫자
    
    + `to` :마칠 숫자
    
    + `by` : 증가량을 설정, 기본 옵션은 1
    
    + `length.out` : 벡터의 크기(갯수)를 지정, `by`를 지정하지 않으면 자동으로 균등하게 증가량을 설정
    
    + `along.with` : 수열을 지정하면 동일한 크기의 벡터를 생성

```r
seq(1, 30)
seq(from = 1, to = 30)
seq(from = 1, to = 30, by = 2)
seq(1, 30, 0.2)
```

`seq(1, 30, length = 10)`을 하게 되면 
$$
c = \frac{b - a}{n - 1} = \frac{30-1}{10-1} = \frac{29}{9} = 3.22222
$$
```r
seq(1, 30, length.out = 10)
```

의 결과와 같다. 주의할 점도 존재한다.
```r
seq(1, 30, by = 2, length = 5)
```

`by`와 `length`를 동시에 사용하지 못함.
```r
x <- seq(1, 30, by = 2)
x
seq(1, 30, along.with = x)
```

15개의 자료가 생성되며 우선순위가 높은 `by` 때문에 끝점 30은 `x`에서 포함되지 않는다. 그리고 `along.with`옵션은 `x`와 갯수를 맞추고 끝점 30을 맞추기위해 증가율(2.071429)을 자동으로 조정한다. 

벡터 생성 함수 `paste()`

- 문자형 벡터의 자료 입력에 유용

- 주어진 벡터의 값을 문자형으로 변환하여 결합

```r
paste(1:5)
as.character(1:5)
paste(1, 1:3, sep = "-")
paste(1:3, 1:3, sep = ",")
paste("Today is ", date())
```

## 2-4. Matrix

행과 열의 2차원 자료 구조이다. 다음의 데이터를 살펴보자.

|연령|성적|
|:--:|:--:|
|10대|52|
|10대|48|
|20대|72|
|30대|72|
|30대|34|
|30대|85|
|40대|40|
|50대|88|
|50대|41|

이는 관측값이 9개, 변수가 2개인 데이터이며 다시 말해, 행이 9개, 변수가 2개인 2차원 행렬이라고도 한다.

행렬을 생성하는 방법을 알아보자. 
```r
m <- matrix(1:9, ncol = 3, nrow = 3)
m
```

기본 옵션은 열부터 차례대로 입력하는 방법이다. 그러나 행부터 입력하고 싶다면
```r
matrix(1:9, ncol = 3, byrow = T)
```

또한 컬럼수를 지정하면 자동으로 행의 수가 지정이 된다.
```r
y <- diag(c(1,3,4))
y
diag(y) # 대각열만 추출
t(m) # 전치행렬
solve(y) # 역행렬
m %*% y # 행렬곱
m * y # 원소 곱
```

주의할 점은 행렬곱을 `*`으로 하게 되면 원소들만 곱해지는 명령어이다.

다음은 선형방정식 풀이에서 행렬을 활용하는 방법이다. 
$$
\begin{split}
\begin{cases}
x + 2y = 10\\
x - y = 1
\end{cases} &\Leftrightarrow \left[\begin{matrix}
1 & 2 \\
1 & -1 \\
\end{matrix}\right]
\left[\begin{matrix}
x \\
y \\
\end{matrix}\right] =
\left[\begin{matrix}
10 \\
1 \\
\end{matrix}\right]\\
&\Leftrightarrow 
\left[\begin{matrix}
x \\
y \\
\end{matrix}\right] =\left[\begin{matrix}
1 & 2 \\
1 & -1 \\
\end{matrix}\right]^{-1}
\left[\begin{matrix}
10 \\
1 \\
\end{matrix}\right]\\
&\Leftrightarrow 
\left[\begin{matrix}
x \\
y \\
\end{matrix}\right] =-\frac{1}{3}\left[\begin{matrix}
-1 & -2 \\
-1 & 1 \\
\end{matrix}\right]
\left[\begin{matrix}
10 \\
1 \\
\end{matrix}\right] = \left[\begin{matrix}4 \\
3\end{matrix}\right]
\end{split}
$$
```r
A <- matrix(c(1, 2, 1, -1), 2, 2, byrow = T)
b <- c(10, 1)
solve(A)%*%b
solve(A, b)
```

## 2-5. Array

배열(Array)는 벡터의 원소들이 벡터로 구성된 형태이며 Matrix와 거의 동일하나 다차원 구조를 정의할 수 있다.

- `array(data, dim = c(행의 갯수, 열의 갯수, 행렬의 갯수)`

    + `data` :  Vector의 자료
    
    + `dim` : 각 차원의 Vector 크기(`c(2, 5, 10)`: 전체는 3차원이고 1차원의 원소는 2, 2차원은 5, 3차원은 10로 총 $2\times 5\times 10 = 100$개)
    
    + `dimnames` : 각 차원의 리스트 이름
    
```r
array(1:6)
array(1:6, dim = c(2, 3))
arr <- array(1:24, dim = c(3, 4, 2))
arr
```

3차원 배열의 원소들을 알고 싶을 때에는
```r
arr[,,1]
arr[,,2]
arr[,2,1]
arr[1,2,1]
```

을 사용한다. 이 때 빈칸으로 두게 되면 전체를 출력한다는 의미이다.
```r
arr1 <- array(1:8, c(2,2,2))
arr2 <- array(8:1, c(2,2,2))
arr1
arr2
arr1 + arr2
arr1 * arr2
arr1 %*% arr2 # 두 배열의 곱의 합
```

## 2-6. List

서로 다른 기본 자료형을 가질 수 있는 자료 구조들의 모임이다. 

형태가 같아야하는 `array`와 달리 다른 구조형식끼리 묶을 수 있다. 
```r
a <- 1:10
b <- 11:15
klist <- list("a" = a, "b" = b, "name" = "example")
length(klist)
klist
klist$b
names(klist)
```

## 2-7. Data frame

- 데이터프레임은 형태(Mode)가 일반화된 행렬이다

- 데이터프레임이라는 하나의 객체에 여라 종류의 자료가 들어갈 수 있음 

- 데이터프레임의 각 열은 각각 변수와 대응

- 분석이나 모형 설정에 적합한 자료 객체

```r
Age1 <- c("10", "10", "20", "30")
Age2 <- c(11, 18, 21, 36)
frame1 <- cbind(Age1, Age2)
is.numeric(Age2)
is.numeric(frame1[,2])
frame2 <- data.frame(age1 = Age1, age2 = Age2)
is.numeric(frame2[,2])
```

## 2-8. Factor

- 벡터 객체 중 범주형 데이터를 원소로 갖는 요인 객체를 생성하는 함수 

- 저장값이 갖는 의미보다 구별하기 위한 값으로 사용(명목형 척도, 순서형 척도)

- `factor(x = character(), levels, labels = levels, exclude = NA, ordered = is.ordered(x), nmax = NA)`

    + `x` : 요인으로 만들 벡터
    
    + `levels` : 주어진 데이터 중 factor의 각 값(수준)으로 할 값을 벡터 형태로 지정
    
    + `labels` : 실제 값 외에 사용할 요인 이름(1:남자 $\rightarrow$ "남자"/"M"으로 변경)
    
    + `exclude` : 요인으로 사용하지 않을 값
    
    + `ordered` : 순서 여부 지정(`TRUE, FALSE`), 순서 있는 범주의 경우 사용하며 순서는 levels에 의해 명시적으로 지정하는 것을 권장
    
    + `nmax` : 최대 level의 수(최대 요인수)
    
```r
x <- c(1, 2, 3, 4, 5)
factor(x, levels = c(1, 2, 3, 4))
factor(x, levels = c(1, 2, 3, 4), labels = c("a", "b", "c", "d"))
factor(x, levels = c(1, 2, 3, 4), ordered = TRUE)
is.factor(x)
is.factor(as.factor(x))
```

`factor()`함수를 이용하면 자료를 범주형으로 인식하게 된다.

# 3. Question

## 3-1. Question

다음의 데이터를 입력하고 범주(연령대)별 평균성적, 총 성적을 계산해 입력하시오.(평균을 일일히 입력하지 마세요, `rep()`활용)

|연령|성적|
|:--:|:--:|
|10대|52|
|10대|48|
|20대|72|
|30대|72|
|30대|34|
|30대|85|
|40대|40|
|50대|88|
|50대|41|

## 3-2. Question

`x1`이라는 숫자형 변수에 1부터 8까지의 정수를 할당하고, `x2`라는 문자형 변수에 알파벳 `A`부터 `H`까지 할당한 데이터 `df`를 만든 다음, 홀수만이 포함된 데이터 `adj.df`를 만드시오.
