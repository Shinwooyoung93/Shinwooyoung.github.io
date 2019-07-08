---
layout: post
title:  "Basic Lecture6"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. R 그래프

- R은 그래프를 위한 다양한 옵션을 제공

- Packages(library)를 이용한 다양한 함수를 구현 가능

- 종류

|함수|설명|
|:--:|:--------------:
|`plot(x)`|x로만 구성되는 1차원 그래프, x축은 index|
|`plot(x, y)`|x와 y로 이루어진 2차원 그래프|
|`pie(x)`|파이차트|
|`boxplot(x)`|box-whiskers 그래프|
|`hist(x)`|수치형 자료에 대한 히스토그램|
|`barplot(x)`|범주형 자료에 대한 히스토그램|
|`qqnorm(x)`|정규분포 하에서 분위수에 대한 그래프|
|`qqplot(x, y)`|x분위수, y분위수에 대한 그래프|

- 기본적으로 x-축과 y-축으로 이루어진 함수를 그릴 때 `plot()`함수를 사용

- `plot(x, y, type, xlim, ylim, xlab, ylab, lty, lwd, main, ...)`

    + `type` : "p"(point), "l"(line), "b"(point + line), "h"(vertival lines), "s"(step), "o"(overlays), "n"(no lines)
    
    + `xlim = c(0, 1), ylim = c(1, 2)` : x축을 (0, 1)사이, y축을 (1, 2)사이로 지정해서 보여줌
    
    + `xlab = "x축", ylab = "y축"` : x축의 이름을 "x축", y축의 이름을 "y축"으로 지정해서 보여줌
    
    + `lty = 1, 2, 3, 4, 5, 6` : 선의 유형(실선, 점선)을 지정
    
    + `lwd = 1, 2, 3` : 선의 굵기를 지정
    
    + `main = "Title"` : 그래프의 제목(타이틀)을 "Title"로 지정
    
- `plot()`을 그린 후에 겹쳐 그리는 데 다음의 함수들을 사용할 수 있음

|함수|설명|
|:--:|:--:|
|`points(x, y)`|추가적인 포인트를 추가, plot함수와 매우 유사|
|`lines(x, y)`|라인을 추가하는 함수|
|`text(x, y, labels)`|해당 점(x, y)에 labels를 추가함|
|`segments(x0, y0, x1, y1)`|점(x0, y0)와 점(x1, y1)을 잇는 선을 추가|
|`abline(a, b)`|a절편과 b기울기를 갖는 직선을 추가|
|`abline(h = y)`|수평선(horizontal line)을 추가|
|`abline(v = x)`|수직선(vertical line)을 추가|

- 그래프를 그릴 때 유용한 함수 몇가지가 존재

    + `par(mfrow = c(2, 3))` : 한개의 사이즈가 들어가는 그래프 안에 2행 3열의 격자를 만들고 그 안에 그래프들을 추가
    
    + `legend()` : 그래프 안에 범례를 넣는 함수

## 1-1. 범주형 자료

이 절에서는 다음의 빈도표를 이용해 시각화를 진행하겠다.

|범주|A|B|C|D|
|:--:|:--:|:--:|:--:|:--:|
|빈도|10|30|20|40|

```r
freq <- c(10, 30, 20, 40)
label<- c("A", "B", "C", "D")
rbind("범주" = label,
      "빈도" = freq)
```

### 1-1-1. Bar plot

막대그래프를 그리는 방법이다.
```r
barplot(freq, main = "막대그래프", names.arg = label)
```

<center><img src="/assets/Basic_lecture6/2.png"></center>

여기에 막대그래프의 색을 추가해보자. 

색은 범주의 갯수만큼 줘도 되고 전체를 하나의 색으로 통일하고 싶다면 번호 1개만 입력하면 된다.
```r
par(mfrow = c(1, 2))
barplot(freq, main = "막대그래프 색 없음", names.arg = label)
barplot(freq, main = "막대그래프 색 있음", names.arg = label, col = c(2, 3, 4, 5))
```

<center><img src="/assets/Basic_lecture6/3.png"></center>

### 1-1-2. Pie plot

다음은 원형 차트이다. 
```r
par(mfrow = c(2, 2))
pie(freq, main = "Pie chart", labels = label)
pie(freq, main = "Pie chart with large radius", labels = label, radius = 1)
pie(freq, main = "Pie chart with clockwise", labels = label, clockwise = T)
percentile <- round(100*freq/sum(freq), 1)
pie.percent<- paste(percentile, "%", sep = "")
color <- c("grey90", "grey50", "black", "grey30")
pie(freq, main = "Adjusted pie chart", labels = pie.percent, clockwise = T, col = color)
legend("topright", legend = label, cex = 0.8, fill = color)
```

<center><img src="/assets/Basic_lecture6/4.png"></center>

## 1-2. 수치형 자료

이 절에서는 다음의 자료를 이용해 시각화를 진행하겠다.

|x|y|
|:--:|:--:|
|0|sin(0)|
|-2|sin(-2)|
|-1|sin(-1)|
|1|sin(1)|
|2|sin(2)|
|3|sin(3)|
|4|sin(4)|

### 1-2-1. 산점도

주의할 것은 위에서 사용한 `par()`옵션 때문에 뒤에 1개의 그래프를 그리고 싶을 때 격자의 형태로 생기게 된다. 

따라서 항상 초기화해주는 습관이 중요하다.
```r
par(new = FALSE)
x <- c(0, -2, -1, 1, 2, 3, 4)
y <- sin(x)
plot(x, y, main = "Scatter plot")
```

<center><img src="/assets/Basic_lecture6/5.png"></center>

다음은 선으로 연결해보자.
```r
par(mfrow = c(2, 2))
plot(x, y, type = "l", main = "Wrong line plot")
data <- data.frame(x = x, y = y)
sort.data <- data[order(data$x),]
plot(sort.data$x, sort.data$y, type = "l", main = "Line plot", xlab = "x", ylab = "y")
plot(sort.data$x, sort.data$y, type = "l", main = "Color plot", xlab = "x", ylab = "y", col = "blue")
plot(sort.data$x, sort.data$y, type = "l", main = "Line plot with grid", xlab = "x", ylab = "y", col = "black", lwd = 2, ylim = c(-1.5, 1.5))
grid <- seq(-2, 4, by = 0.01)
grid.val <- sin(grid)
points(grid, grid.val, col = "blue", lwd = 2, type = "l")
```

<center><img src="/assets/Basic_lecture6/6.png"></center>

### 1-2-2. Histogram

- 수치형벡터 1개만이 필요한 그래프

- `hist(x, breaks = "Sturges", freq = NULL, probabilty = !freq, col = NULL, main = NULL, ylim = NULL, ...)`

    + `x` : 히스토그램을 위한 벡터 데이터
    
    + `breaks` : 계급 구간의 수, `Sturges`는 데이터가 정규분포를 가정
    
    + `freq` : `TRUE`는 빈도 수, `FALSE`는 상대도수를 나타냄
    
    + `probabilty` : `freq`의 반대
    
```r
par(mfrow = c(2, 2))
hist(x, main = "Raw histogram")
hist(x, main = "Break histogram", breaks = 5)
hist(x, main = "Color histogram", breaks = 5, col = c("red", "orange", "yellow", "green", "blue"))
hist(x, main = "Color histogram", probability = T, breaks = 6)
points(density(x), col = "red", type = "l", lwd = 2)
```

<center><img src="/assets/Basic_lecture6/7.png"></center>

`density()`함수를 조금 더 자세히 살펴보자.
```r
par(mfrow = c(1, 1))
data(quakes) # 지진강도의 데이터 셋 불러오기
head(quakes)
hist(quakes$mag, main = "지간발생강도의 분포", xlab = "지진강도", ylab = "발생건수", freq = F)
points(density(quakes$mag), col = "red", type = "l", lwd = 2)
```

<center><img src="/assets/Basic_lecture6/8.png"></center>

이처럼 히스토그램의 분포를 선으로 나타낼 수 있다.

## 1-3. 통계 그래프

정규분포의 형태는 아래의 그림과 같다.
```r
x <- seq(-3, 3, by = 0.01)
y <- dnorm(x, mean = 0, sd = 1)
plot(x, y, main = "Normal distribution", xlab = "x", ylab = "probabilty", type = "l", lwd = 2)
```

<center><img src="/assets/Basic_lecture6/9.png"></center>

### 1-3-1. Q-Q plot

이러한 데이터의 정규성을 알 수 있는 방법은 여러가지가 있지만 그 중 강력한 검증도구 중 하나가 Q-Q plot이다.
```r
par(new = F)
qqnorm(quakes$mag, main = "Q-Q plot")
qqline(quakes$mag, lwd = 2)
```

<center><img src="/assets/Basic_lecture6/10.png"></center>

위에서 봤던 그림처럼 이는 정규분포를 따르지 않는다. 이번에는 정규분포가 따르는 데이터를 테스트 해보자.
```r
rm(list = ls())
set.seed(1)
x <- rnorm(100)
qqnorm(x, main = "Q-Q plot for normal distribution")
qqline(x, lwd = 2)
```

<center><img src="/assets/Basic_lecture6/11.png"></center>

정규분포를 따른다는 것을 쉽게 파악할 수 있다.

### 1-3-2. Classification plot

먼저 2차원의 원본데이터(`iris` 데이터)를 살펴보자.

<center><img src="/assets/Basic_lecture6/1.png"></center>

```r
data(iris)
str(iris)
dim(iris)
```

`iris` 데이터를 `Species`변수가 `setosa`와 같은 데이터, 다른 데이터 2개로 만들자.
```r
data1 <- iris[iris$Species=="setosa",]
data2 <- iris[iris$Species!="setosa",]
```

```r
plot(x = data1$Sepal.Length, y = data1$Sepal.Width, xlim = c(4, 8), ylim = c(2, 4.5), main = "Iris data", xlab = "Sepal.Length", ylab = "Sepal.Width", col = "red")
points(x = data2$Sepal.Length, y = data2$Sepal.Width, col = "blue")
legend("topleft", legend = c("setosa", "versicolor"), cex = 0.8, col = c("red", "blue"), lwd = 2)
```

<center><img src="/assets/Basic_lecture6/12.png"></center>

그래프는 다음과 같으며 2개의 범주를 잘 나눌 수 있는 직선을 그리는 것은 통계학에서 매우 중요한 문제이다. 

이를 잘 설명할 수 있는 직선은 (4, 2)와 (7.5, 5)를 잇는 직선으로 기대된다.
```r
plot(x = data1$Sepal.Length, y = data1$Sepal.Width, xlim = c(4, 8), ylim = c(2, 4.5), main = "Classfication iris data", xlab = "Sepal.Length", ylab = "Sepal.Width", col = "red")
points(x = data2$Sepal.Length, y = data2$Sepal.Width, col = "blue")
legend("topleft", legend = c("setosa", "versicolor"), cex = 0.8, col = c("red", "blue"), lwd = 2)
segments(x0 = 4, y0 = 2, x1 = 7.5, y1 = 5, col = "green", lty = 2, lwd = 2)
```

<center><img src="/assets/Basic_lecture6/13.png"></center>

### 1-3-3. Box-plot

- 주로 범주별 수치형 변수들의 차이가 있는지 보고싶을 때 사용하는 그래프

- `boxplot(formula, data, ...)`

    + `formula` : 2가지 종류가 존재
        
        + 변수 1개 : 변수 1개에 대해 상자그림을 그림
        
        + 독립변수(수치형) ~ 종속변수(범주형) : x축에 종속변수(범주)들이 표기되고 각각 수치형변수들에 대해 상자그림을 그림
   
   
변수 1개일 때를 보자.
```r
boxplot(iris$Sepal.Length, main = "Sepal length boxplot")
```

<center><img src="/assets/Basic_lecture6/14.png"></center>

다음은 독립변수 ~ 종속변수 형태로 살펴보자.
```r
boxplot(iris$Sepal.Length ~ iris$Species, main = "Sepal length boxplot")
```

<center><img src="/assets/Basic_lecture6/15.png"></center>

# 2. 3차원 그래프

라이브러리 없이 그릴 수 있는 3차원 그래프에 대해서 살펴보자.

- `persp(x, y, z, theta = , ..., phi = , ..., border)`

    + `z` : z축을 설정, (x[i], y[j])에서의 높이가 z[i, j]로 저장된 행렬
    
    + `theta` : 그림의 좌우 회전 각도
    
    + `phi` : 그림의 상하 회전 각도
    
    + `border` : 조감도의 각 면의 테두리선에 사용할 색을 설정, NA면 테두리선을 그리지 않음

$$
\begin{cases}
\begin{split}
X \sim U(-2, 2) \\
Y \sim U(-2, 2)
\end{split}
\end{cases} \Rightarrow \phi(x, y) = \frac{1}{2\pi}\exp\left(-\frac{x^2 + y^2}{2}\right)
$$

```r
par(mfrow = c(2, 2))
x <- seq(-2, 2, by = 0.05)
y <- seq(-2, 2, by = 0.05)
z <- matrix(NA, nrow = length(x), ncol = length(y))

for(i in 1:length(x)){
  for(j in 1:length(y)){
    z[i, j] <- 1/(2*pi)*exp(-(x[i]^2 + y[j]^2)/2)
  }
}
persp(x = x, y = y, z = z, phi = 0, theta = 0, expand = 0.5, main = "phi = 0, theta = 0")
persp(x = x, y = y, z = z, phi = 0, theta = 30, expand = 0.5, main = "phi = 0, theta = 30")
persp(x = x, y = y, z = z, phi = 30, theta = 0, expand = 0.5, main = "phi = 30, theta = 0")
persp(x = x, y = y, z = z, phi = 30, theta = 30, expand = 0.5, main = "phi = 30, theta = 30")
```

<center><img src="/assets/Basic_lecture6/16.png"></center>

그러나 위의 3차원 그래프는 원하는 값의 위치를 알기 쉽지는 않다. 

따라서 다음 강의에서 다룰 `ggplot()`함수들을 사용해 추후에 방법을 설명하도록 하겠다.

# 3. Question

## 3-1. Question

`iris`데이터를 이용해 다음의 그래프 분석을 실시하시오.(`data(iris)`활용)

<center><img src="/assets/Basic_lecture6/17.png"></center>

## 3-2. Question

`quakes`데이터를 이용하되 `mag`를 4.5미만, 4.5이상 5미만, 5.0이상으로 분류(범주는 `1`, `2`, `3`)하여 다음의 그래프 분석을 실시하시오.(`data(quakes)`활용)

<center><img src="/assets/Basic_lecture6/18.png"></center>
