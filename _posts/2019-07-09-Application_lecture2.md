---
layout: post
title:  "Application Lecture2"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. Review

## 1-1. Review lecture

`ggplot2`라이브러리를 이용해 고급시각화를 진행하였다. 

크게 2단계로 이루어졌다.

1. `ggplot()`함수를 이용해 데이터가 그려질 공간을 정의

2. `geom`옵션들을 이용해 시각화를 진행

단순히 `ggplot()`함수만을 사용하는 것은 의미가 없었다.
```r
rm(list = ls())
library(ggplot2)
data(iris)
p <- ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width))
```

따라서 공간 위에 점을 찍는 방법은 아래와 같았다.
```r
q <- geom_point(aes(colour=Species), size = 2)
p + q
```

<center><img src="/assets/Application_lecture2/1.png"></center>

또한 제목들을 변경하는 코드는 아래와 같았다.
```r
l <- labs(x = "Sepal Length", y = "Sepal Width", title = "Scatter plot iris data")
p + q + l
```

<center><img src="/assets/Application_lecture2/2.png"></center>

범주형 자료에 대해 `melt()`함수를 이용해 막대그래프를 그리는 방법은 아래와 같다.
```r
library(reshape)
new.iris <- melt(iris, id.vars = "Species")
head(new.iris)
```
```r
##   Species     variable value
## 1  setosa Sepal.Length   5.1
## 2  setosa Sepal.Length   4.9
## 3  setosa Sepal.Length   4.7
## 4  setosa Sepal.Length   4.6
## 5  setosa Sepal.Length   5.0
## 6  setosa Sepal.Length   5.4
```
```r
p <- ggplot(new.iris, aes(x = Species, y = value, fill = variable))
b <- geom_bar(stat = "identity")
l <- labs(title = "Bar plot iris data")
p + b + l
```

<center><img src="/assets/Application_lecture2/3.png"></center>

끝으로 수치형 자료에 대해 히스토그램을 그리는 방법은 아래와 같다.
```r
p <- ggplot(iris, aes(x = Sepal.Width))
h <- geom_histogram(binwidth=0.2, color = "black", fill = "tomato", aes(y=..density..))
d <- geom_density(fill = "blue", alpha = 0.3)
l <- labs(x = "Sepal Width", title = "Histogram iris data")
p + h + d + l
```

<center><img src="/assets/Application_lecture2/4.png"></center>

## 1-2. Another topic

우리나라 지도 위에 그래프를 그리는 방법에 대해 배워보자. 우선 우리나라의 지도를 한번 그려보도록 하자.

한국의 위도와 경도는 `경도(lng) = 128.25`, `위도(lat) = 35.95`이다.
```r
# install.packages("leaflet")
rm(list = ls())
library(ggplot2)
library(leaflet)
m <- leaflet() %>%
  addTiles() %>%
  setView(lng=128.25, lat=35.95, zoom = 7)
m
```

<center><img src="/assets/Application_lecture2/5.png"></center>

여기 위에 우리나라의 와이파이가 어떻게 위치해있는지 점을 찍어보고자 `wifi.csv`파일을 불러온다.
```r
wifi <- read.csv(file.choose())
head(wifi)
```
```r
##   company      lat      lon
## 1      KT 37.74417 128.9056
## 2      KT 37.72806 128.9543
## 3      KT 37.75710 128.8900
## 4      KT 37.74769 128.8840
## 5      KT 37.74866 128.9073
## 6      KT 37.74281 128.8827
```

이는 `한국통신사업자연합회(KOTA)`에서 공공와이파이 위치를 담은 데이터이다. 이를 바탕으로 `addMarkers()`함수를 이용해 점을 찍어보자.
```r
p <- m %>% addMarkers(data=wifi, lng= ~lon, lat= ~lat, clusterOptions = markerClusterOptions())
p
```

<center><img src="/assets/Application_lecture2/6.png"></center>

이런식으로 우리나라 지도에 공공와이파이 위치가 찍힌 그래프를 확인할 수 있다.

그러나 통신사의 정보를 추가해주기 위해 색을 덧칠한다.
```r
pal <- colorFactor(c("red", "orange", "navy"), domain = unique(wifi$company))
a <- m %>% addCircles(data = wifi, lng= ~lon, lat= ~lat, 
                      opacity = 2, weight = 4, color = ~pal(company)) %>%
  addLegend(data = wifi, pal = pal, values = ~company,
            opacity = 0.7, title = "Wifi via company")
a
```

<center><img src="/assets/Application_lecture2/7.png"></center>

# 2. Apply real data

## 2-1. Crime data

먼저 데이터를 불러와보자.
```r
rm(list = ls())
library(ggplot2)
library(reshape)
library(leaflet)
crime <- read.csv(file.choose())
```

데이터의 구조는 다음과 같이 이루어져 있다.

|Variable|Type|Summary|
|:---:|:---:|:---:|
|Category|Factor|범죄 종류|
|DayOfWeek|Factor|범죄 발생 요일|
|PdDistrict|Factor|경찰서 이름|
|X|Numeric|경도|
|Y|Numeric|위도|
|dist|Numeric|사건지점으로부터 경찰서 거리|
|time|Factor|사건해결까지 소요시간|

```r
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

먼저 `leaflet()`을 이용해 시각화를 진행한다.
```r
m <- leaflet() %>%
  addTiles() %>%
  setView(lng=-122.43317, lat=37.76345, zoom = 12)
p <- m %>% addMarkers(data=crime, lng= ~X, lat= ~Y, clusterOptions = markerClusterOptions())
p
```

<center><img src="/assets/Application_lecture2/8.png"></center>

다음 범주에 따른 시각화를 진행한다.
```r
pal <- colorFactor(c("navy", "red"), domain = unique(crime$Category))
a <- m %>% 
  addCircles(data = crime, lng= ~X, lat= ~Y, 
             opacity = 2, weight = 4, color = ~pal(Category)) %>%
  addLegend(data = crime, pal = pal, values = ~Category,
            opacity = 0.7, title = "True plot")
a
```

<center><img src="/assets/Application_lecture2/9.png"></center>

우측 상단에 `ROBBERY`범주가 많은 것을 확인할 수 있다. 다음의 변수들을 이용해 시각화를 진행한다.

## 2-2. Target variable

우리의 분석 목표 대상인 종속변수인 `Category`의 특징을 살펴보도록 하자.
```r
p <- ggplot(data=crime,aes(x=Category, fill=Category))
b <- geom_bar()
l <- labs(title = "Category variable")
p + b + l
```

<center><img src="/assets/Application_lecture2/10.png"></center>

강도사건이 납치사건에 비해 조금 더 많은 것을 확인할 수 있다.

## 2-3. Numeric variables

수치형 변수들 `dist`와 `time`변수들의 특징을 살펴보도록 하자.
```r
p1 <- ggplot(crime, aes(x = dist))
h1 <- geom_histogram(binwidth=1, color = "black", fill = "tomato", aes(y=..density..))
d1 <- geom_density(fill = "blue", alpha = 0.3)
l1 <- labs(title = "Histogram dist variable")
p1 + h1 + d1 + l1
```

<center><img src="/assets/Application_lecture2/11.png"></center>

데이터의 형태가 봉우리가 2개임을 알 수 있으며, `time`변수의 특징을 살펴보자.
```r
p2 <- ggplot(crime, aes(x = time))
h2 <- geom_histogram(binwidth=1, color = "black", fill = "tomato", aes(y=..density..))
d2 <- geom_density(fill = "blue", alpha = 0.3)
l2 <- labs(title = "Histogram time variable")
p2 + h2 + d2 + l2
```

<center><img src="/assets/Application_lecture2/12.png"></center>

마찬가지로 봉우리가 2개인 데이터의 형태를 갖는다.

끝으로 `dist`와 `time`변수들의 상관관계를 확인해보자.
```r
# install.packages("corrplot")
library(corrplot)
numeric.data <- subset(crime, select = c("dist", "time"))
cor.data <- cor(numeric.data)
corrplot(cor.data, method="number")
```

<center><img src="/assets/Application_lecture2/13.png"></center>

`dist`와 `time`은 상관계수가 0.91로 상관관계가 매우 큰 것으로 판단된다. 

즉 경찰서로부터 거리가 멀면 사건으로부터 해결 소요 시간이 길어진다는 것을 파악할 수 있다.

## 2-4. Factor variables

범주형 변수들인 `DayOfWeek`, `PdDistrict`와 종속변수 `Survived`간의 관계를 살펴보자.
```r
p3 <- ggplot(crime, aes(x = DayOfWeek, fill = Category))
b3 <- geom_bar()
l3 <- labs(title = "Category vs DayOfWeek")
p3 + b3 + l3
```

<center><img src="/assets/Application_lecture2/14.png"></center>

`DayOfWeek`에 따른 범죄종류에는 큰 차이가 있는 것으로 확인되지 않는다.
```r
p4 <- ggplot(crime, aes(x = PdDistrict, fill = Category))
b4 <- geom_bar()
l4 <- labs(title = "Category vs PdDistrict")
p4 + b4 + l4
```
<center><img src="/assets/Application_lecture2/15.png"></center>

`PdDistrict`에 따른 범죄 종류에는 차이가 있는 것으로 판단된다. 끝으로 그래프들을 모아서 그려보면
```r
# install.packages("gridExtra")
library(gridExtra)
grid.arrange(p1 + h1 + d1 + l1,
             p2 + h2 + d2 + l2,
             p3 + b3 + l3,
             p4 + b4 + l4)
```
<center><img src="/assets/Application_lecture2/16.png"></center>

# 3. Question

다음의 형식을 이용해 `wifi.csv`데이터와 `crime.csv`데이터를 이용해 분석 보고서를 완성하시오.

```r
---
title: "EDA report"
author: "SSWU, Your Name"
date: \today
output: html_document
---
```

```r
```{r, echo = T, out.width = '70%', fig.align='center'}
(your script)
``` 
```

### 1. Wifi data
#### 1-1. Plotting data
#### 1-2. Conclusion
### 2. Crime data
#### 2-1. Target variable
#### 2-2. Numeric variables
#### 2-3. Factor variables
#### 2-4. Conclusion
### 3. Conclusion
