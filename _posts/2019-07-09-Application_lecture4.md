---
layout: post
title:  "Application Lecture4"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. Clustering

## 1-1. 정의 및 목적

- 유사한 속성들을 갖는 관측치들을 묶어 전체 데이터를 몇 개의 군집(그룹)으로 나누는 것

<center><img src="/assets/Application_lecture4/4.png"></center>

- 군집화의 기준은 2가지

    + 동일한 군집에 소속된 관측치들은 서로 유사할수록 좋음 $\Rightarrow$ 군집내 유사도 최대화 
    
    + 상이한 군집에 소속된 관측치들은 서로 다를수록 좋음 $\Rightarrow$ 군집간 유사도 최소화 
    
- 분류와 군집의 차이점

    + 군집화(Clustering) : 사전 정의된 범주가 없는 데이터에서 최적의 그룹을 찾아가는 문제
    
    + 분류(Classification) : 사전 정의된 범주가 있는 데이터로부터 모델링을 하는 문제
    
<center><img src="/assets/Application_lecture4/1.png"></center>

- 군집화의 적용사례

    + 타겟 마케팅
    
    + 유사문서 군집화
    
    + 오존농도 패턴 군집화
    
<center><img src="/assets/Application_lecture4/2.png"></center>
<center><img src="/assets/Application_lecture4/3.png"></center>

## 1-2. Simple example

`k-means`군집화를 실시할 때, `kmeans()`함수를 사용한다.

- `kmeans(x, centers, ...)`

    + `x` : 군집화를 위해 필요한 수치형 데이터
    
    + `centers` : 군집화를 할 갯수
  
하지만 군집의 갯수를 설정하는 것은 항상 어려움이 존재한다. 보통은 사전지식을 많이 이용한다.
```r
rm(list = ls())
library(ggplot2)
set.seed(1)
data(iris)
p <- ggplot(iris, aes(Sepal.Length, Sepal.Width))
s<- geom_point(aes(color = Species))
l<- labs(title = "True plot")
p + s + l
```

<center><img src="/assets/Application_lecture4/5.png"></center>

다음의 데이터를 사용하고, `kmeans`함수를 이용해 군집화를 실시한다.

그를 위해 수치형 데이터 `numeric.data`를 생성한다.
```r
numeric.data <- subset(iris, select = c("Sepal.Length", "Sepal.Width"))
fit <- kmeans(numeric.data, centers = 3)
clust<-paste0(fit$cluster, "-th clust")
table(clust)
```
```r
## clust
## 1-th clust 2-th clust 3-th clust 
##         53         47         50
```

이를 이용해 군집화가 완료된 데이터 `df`를 생성하고 시각화를 진행한다.
```r
df <- data.frame(iris, cluster = clust)
head(df)
```
```r
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species    cluster
## 1          5.1         3.5          1.4         0.2  setosa 3-th clust
## 2          4.9         3.0          1.4         0.2  setosa 3-th clust
## 3          4.7         3.2          1.3         0.2  setosa 3-th clust
## 4          4.6         3.1          1.5         0.2  setosa 3-th clust
## 5          5.0         3.6          1.4         0.2  setosa 3-th clust
## 6          5.4         3.9          1.7         0.4  setosa 3-th clust
```
```r
p <- ggplot(df, aes(Sepal.Length, Sepal.Width))
s1<- geom_point(aes(color = Species))
s2<- geom_point(aes(color = clust))
l1<- labs(title = "True plot")
l2<- labs(title = "Cluster plot")
library(gridExtra)
grid.arrange(p + s1 + l1,
             p + s2 + l2)
```

<center><img src="/assets/Application_lecture4/6.png"></center>

군집화를 실시한 그래프는 위와 같으며 사전에 정의된 범주와 유사하게 나온 것을 확인할 수 있다.

# 2. Apply real data

## 2-1. Clustering data

```r
rm(list = ls())
library(ggplot2)
library(reshape)
library(leaflet)
set.seed(2)
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

```r
numeric.data <- subset(crime, select = c("dist", "time"))
scale.data <- scale(numeric.data)
head(scale.data)
```
```r
##        dist      time
## 1 0.6041134 0.7406374
## 2 0.4675987 0.9220892
## 3 1.3458032 0.7438946
## 4 0.9444833 0.8133530
## 5 0.7020156 0.6647243
## 6 0.6485337 0.8117250
```
```r
fit <- kmeans(scale.data, centers = 2, iter.max = 100)
pred<- paste0(fit$cluster, "-th clust")
df  <- data.frame(scale.data, Category = crime$Category, pred = pred, X = crime$X, Y = crime$Y)
p <- ggplot(data = df, aes(x = dist, y = time))
s1<- geom_point(aes(color = Category))
s2<- geom_point(aes(color = pred))
l1<- labs(title = "True plot")
l2<- labs(title = "K-means plot")

library(gridExtra)
grid.arrange(p + s1 + l1,
             p + s2 + l2)
```

<center><img src="/assets/Application_lecture4/7.png"></center>

`k-means`를 통해 그려진 군집화는 실제 데이터처럼 분류를 보장할 수 있는 것은 아니다. 

`Category`변수를 신경쓰지 않고 만들어진 모델링이기 때문에, 성능을 당연히 보장할 수 없으며 분리된 군집이 `Category`의 어떤 부분인지 알 수 없다.

추가적으로 본래의 `Category`변수와 `clust`변수를 이용해 지도에서의 시각화를 진행해보자.
```r
m <- leaflet() %>%
  addTiles() %>%
  setView(lng=-122.43317, lat=37.76345, zoom = 12)
pal1 <- colorFactor(c("navy", "red"), domain = unique(df$Category))
pal2 <- colorFactor(c("purple", "orange"), domain = unique(df$pred))

a1 <- m %>% 
  addCircles(data = df, lng= ~X, lat= ~Y, 
             opacity = 2, weight = 4, color = ~pal1(Category)) %>%
  addLegend(data = df, pal = pal1, values = ~Category,
            opacity = 0.7, title = "True plot") 
a1
```

<center><img src="/assets/Application_lecture4/8.png"></center>

```r
a2 <- m %>% 
  addCircles(data = df, lng= ~X, lat= ~Y, 
             opacity = 2, weight = 4, color = ~pal2(pred)) %>%
  addLegend(data = df, pal = pal2, values = ~pred,
            opacity = 0.7, title = "Clust plot")
a2
```

<center><img src="/assets/Application_lecture4/9.png"></center>

`k-means`를 통해 지역간의 차이를 확인할 수 있고 `purple`, `orange`색은 서로 비슷한 지역이라고 판단한다.

## 2-2. Plotting via model

이제 `k-means`를 통해 분리된 군집들의 세부특징을 살펴보자.

이를 위해 `cluster.data`를 정의한다.
```r
cluster.data <- data.frame(crime, cluster = pred)
head(cluster.data)
```
```r
##         Category DayOfWeek PdDistrict         X        Y     dist     time
## 1        ROBBERY       Thu          H -122.4079 37.78151 17.75811 36.78203
## 2        ROBBERY       Sun          H -122.4034 37.77542 16.94264 38.42483
## 3        ROBBERY       Sat          B -122.4043 37.78792 22.18855 36.81152
## 4 MISSING PERSON       Thu          H -122.4081 37.78399 19.79129 37.44037
## 5        ROBBERY       Wed          E -122.4226 37.79264 18.34292 36.09475
## 6        ROBBERY       Fri          H -122.4185 37.77590 18.02345 37.42563
##      cluster
## 1 2-th clust
## 2 2-th clust
## 3 2-th clust
## 4 2-th clust
## 5 2-th clust
## 6 2-th clust
```

### 2-2-1. Target variable

군집에 따른 `Category`변수의 특징을 살펴보자.
```r
p1 <- ggplot(cluster.data, aes(x = Category, fill = cluster))
b1 <- geom_bar()
l1 <- labs(title = "k-means with Category")
p1 + b1 + l1
```

<center><img src="/assets/Application_lecture4/10.png"></center>

`MISSING PERSON`범주에는 1번 집단이 많이 분포하며 `ROBBERY`범주에는 2번 집단이 많이 분포하는 것을 확인할 수 있다.

### 2-2-2. Numeric variable

다음은 군집에 따른 `dist`, `time`변수의 특징을 살펴보자.
```r
p2 <- ggplot(cluster.data, aes(x = dist, color = cluster))
h2 <- geom_histogram(binwidth=1, color = "black", aes(fill = cluster, y=..density..))
l2 <- labs(title = "Histogram dist variable with cluster")
p2 + h2 + l2
```

<center><img src="/assets/Application_lecture4/11.png"></center>

군집에 따른 수치형변수 `dist`는 차이가 존재하며 2번 군집이 오른쪽에 위치하는 것을 확인할 수 있다.
```r
p3 <- ggplot(cluster.data, aes(x = time, color = cluster))
h3 <- geom_histogram(binwidth=1, color = "black", aes(fill = cluster, y=..density..))
l3 <- labs(title = "Histogram time variable with cluster")
p3 + h3 + l3
```
<center><img src="/assets/Application_lecture4/12.png"></center>

군집에 따른 수치형변수 `time`는 차이가 존재하며 2번 군집이 오른쪽에 위치하는 것을 확인할 수 있다.

### 2-2-3. Factor variable

군집에 따른 `Factor`변수의 특징을 살펴보자.
```r
p4 <- ggplot(cluster.data, aes(x = DayOfWeek, fill = cluster))
b4 <- geom_bar()
l4 <- labs(title = "DayOfWeek with cluster")
p4 + b4 + l4
```

<center><img src="/assets/Application_lecture4/13.png"></center>

군집에 따른 범주형변수 `DayOfWeek`는 차이가 존재하지 않으며 `PdDistrict`변수를 살펴보자.
```r
p5 <- ggplot(cluster.data, aes(x = PdDistrict, fill = cluster))
b5 <- geom_bar()
l5 <- labs(title = "PdDistrict with cluster")
p5 + b5 + l5
```

<center><img src="/assets/Application_lecture4/14.png"></center>

군집에 따른 범주형변수 `PdDistrict`는 차이가 존재하는 것으로 보이며 끝으로 그래프들을 모아서 그려보면
```r
library(gridExtra)
grid.arrange(p2 + h2 + l2,
             p3 + h3 + l3,
             p4 + b4 + l4,
             p5 + b5 + l5)
```

<center><img src="/assets/Application_lecture4/15.png"></center>

# 3. Question

다음의 형식을 이용해 `crime.csv`데이터를 이용해 분석 보고서를 완성하시오.

```r
---
title: "Clustering report via crime data"
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
### 2. Clustering
#### 2-1. Modeling
#### 2-2. Plotting via model
### 3. Conclusion
