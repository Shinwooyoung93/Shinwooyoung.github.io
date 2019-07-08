---
layout: post
title:  "Basic Lecture7"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. R 패키지

- R은 기본기능만으로도 많은 작업을 수행할 수 있지만 다음의 기능들을 수행할 경우가 많음

    + 편안한 자료의 정리
    
    + 편안한 데이터의 분석
    
    + 고급 시각화
    
    + 머신러닝/딥러닝
    
- 개발자가 개발한 분석방법을 자유롭게 다운받아 사용할 수 있는 패키지(package)를 제공

- 전 세계의 연구자에 의해 최적화가 잘 되어있으며 빠르고 안정적으로 작동

## 1-1. 패키지 설치

원하는 패키지 명을 다음과 같이 입력한다.
```r
#install.packages("ggplot2")
```

그 후에 다음의 결과를 확인할 수 있다.

<center><img src="/assets/Basic_lecture7/0.PNG" width="300" height="250"></center>

설치만했을 뿐 사용하겠다는 명령어는 입력하지 않았다.

따라서 다음의 명령어를 입력한다.
```r
library(ggplot2)
```

만약 패키지를 사용하고 싶지 않다면 다음의 명령어를 이용해 내재 되어 있는 패키지를 먼저 확인한다.
```r
search()
```

나머지는 R의 기본 패키지이며 우리가 사용하고 싶지 않은 `package:ggplot2`을 다음과 같이 입력한다.
```r
detach("package:ggplot2")
```

또한 패키지를 삭제하는 방법도 있는데 다음과 같이 입력한다.
```r
#remove.packages("ggplot2")
```

`tensorflow`, `randomForest` 등과 같은 많은 패키지들이 존재한다.

# 2. ggplot

- 뉴질랜드 태생의 통계학자이며 R-studio의 Chief Scientist인 Hadely Wickham이 개발

- `ggplot`은 grammer of graphics plot의 약자이며, 데이터를 이해하는데 좋은 시각화 도구

- 데이터 시각화에 대한 요구가 계속해서 증대되고 있으며, `ggplot`은 데이터 시각화를 위한 그래픽 라이브러리로써 현재까지 가장 많이 사용됨

- `ggplot`(=grammer of graphic)의 약자와 같이 grammer 기반의 명령어를 제공

- 문장 형성을 통해 사용자는 데이터 시각화가 가능

- 지속적인 update와 커뮤니티를 통한 우수한 사례 공유가 가능

## 2-1. 원리 및 용어

1. `ggplot` : 메인 함수로 데이터 셋과 표현할 데이터 변수명을 정의한다. 단, 그래프를 그리는게 아니라 정의만 하는 역할이다.

    - `data` : data frame 형태로 저장된 데이터이다.

2. `geoms` : 데이터를 나타내는 기하학적(도식화) 도형의 설명 부분이다. (points, lines, polygons, ...)

    - `geom_point()`, `geom_bar()`, `geom_density()`, `geom_line`, ...
    
    - `aesthetics` : 데이터의 시각적인 요소에 대한 설명(position, size, color, shape, ...) 형태, 투명도, 색상, 라인의 형태 등을 정의
    
    - `scales` : 데이터의 값을 표현하기 위해 각각의 시각적인 요소들을 어떻게 변환해서 나타낼 것인가의 설명 부분이다.
    
    - `stats` : 요약 데이터를 통계적으로 어떻게 변환해서 보여줄 것인지를 설명한다. (counts, means, ...)
    
    - `facets` : 데이터를 어떻게 더 작은 하위 집합으로 나누어서 여러 개의 세부적인 그래프로 보여줄 것인지에 대한 설명 부분

## 2-2. ggplot2의 활용 이해

### 2-2-1. 기초 구조의 이해

`iris` 데이터를 이용해 가로축은 `Sepal.Length`, 세로축은 `Sepal.Width`로 설정해준다.
```r
rm(list = ls())
library(ggplot2)
data(iris)
ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width)) + geom_point()
```

<center><img src="/assets/Basic_lecture7/1.png" width="500" height="400"></center>

`ggplot` 함수 안에서 데이터와 변수를 정의한다. 이 부분은 일종의 백지 위에 그림을 그릴 대상을 정의하는 부분이다. 

즉, 데이터 시각화를 위한 셋팅 부분이다.

그 이후 어떤 시각화를 할지의 구체적인 데이터 시각화 컴포넌트 대상, 통계모형 등을 백지 상에 레이어(layer) 형태로 그림을 그리면 된다.
```r
sepal <- ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width))
sepal + geom_point()
sepal + geom_line()
```

<center><img src="/assets/Basic_lecture7/2.png" width="500" height="400"></center>
<center><img src="/assets/Basic_lecture7/3.png" width="500" height="400"></center>

## 2-3. 다양한 데이터 시각화 컴포넌트

### 2-3-1. Scatter plot

산점도는 `geom_point`를 이용하여 표현할 수 있다.
```r
ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width)) + geom_point(size = 4)
```

<center><img src="/assets/Basic_lecture7/4.png" width="500" height="400"></center>

또한 point의 크기를 이용하여 하나의 그래프에서 확인할 수 있는 정보량을 늘려줄 수 있다.
```r
ggplot(iris, aes(Sepal.Length, Sepal.Width)) +  geom_point(aes(size=Petal.Width))
```

<center><img src="/assets/Basic_lecture7/5.png" width="500" height="400"></center>

`iris`의 종류(Species)별로 다른 색상을 적용해줄 수 있다.
```r
# case 1
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) + geom_point(size = 2)
# case 2
ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_point(aes(colour=Species),size = 2)
```

<center><img src="/assets/Basic_lecture7/6_1.png" width="500" height="400"></center>
<center><img src="/assets/Basic_lecture7/6_2.png" width="500" height="400"></center>

`ggplot`에 넣어도 되고 `geom_point`에 넣어도 상관없어 보이지만 위의 두 케이스는 다르다고 볼 수 있다.

두 식에 `geome_line()`을 추가하면 어떻게 달라지는지 살펴보자.
```r
# case 1
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) + geom_point(size = 2) + geom_line()
```

<center><img src="/assets/Basic_lecture7/7.png" width="500" height="400"></center>

첫번째 식은 `ggplot`내에서 `Species`별로 색상을 다르게 하도록 지정해주었기 때문에 `geom_line()`에서는 각 `Species`별로 선을 그려주었다.

```r
# case 2
ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_point(aes(color=Species), size = 2) + geom_line()
```

<center><img src="/assets/Basic_lecture7/8.png" width="500" height="400"></center>

그러나 두번째는 `ggplot`내에서 색상을 지정해준게 아니라 `geom_point()`에서 지정해주었기 때문에, `geom_line`을 설정하면 전체 데이터로 선을 그려준다.

### 2-3-2. Bar plot

- Bar plots를 그릴때 x축에 대한 y축의 값을 1) count와 2) column의 값으로 표현

    - 1) `geom_bar`은 기본설정이 `stat=“count”`으로 되어있으므로 **각 집단에 속한 사전개수(빈도수)**를 센다.
    
    - 2) 빈도수 대신 **값 자체**를 가지고 막대 그래프를 그릴 땐 `stat=“identity”`를 사용한다. (y값들을 x값들과 1:1로 인식)

-  이산형 데이터는 개별 문자로 표시되는 데이터이며, 값이 연속적으로 변화하지 않고 불연속적

|x축|y축 값의 높이|데이터 시각화 방법|
|:--:|:--:|:--:|
|연속형(numeric)|Count|Histogram|
|이산형(factor)|Count|Bar plot|
|연속형(numeric)|Value|Bar plot|
|이산형(factor)|Value|Bar plot|

```r
ggplot(iris, aes(Species)) + geom_bar(stat = "count")
```

<center><img src="/assets/Basic_lecture7/9.png" width="500" height="400"></center>

`geop_bar(stat = "count")`를 사용하고 싶다면 y값은 입력해서는 안된다. `geop_bar(stat = "identity")`를 사용하게 되면 `Sepal.Length`값들의 합이 나오게 된다.

```r
ggplot(iris, aes(Species, Sepal.Length)) + geom_bar(stat = "identity")
```

<center><img src="/assets/Basic_lecture7/10.png" width="500" height="400"></center>

`fill`을 `Species`로 지정했기 때문에 `Species`에 따라 데이터가 다른 색으로 표시된다.
```r
ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) + geom_bar(stat = "identity")
```

<center><img src="/assets/Basic_lecture7/11.png" width="500" height="400"></center>

`guides(fill=FALSE)`나 `scale_fill_discrete(guide=FALSE)`를 사용하면 오른쪽의 범례를 표시하지 않는다. 

`theme(legend.position=“none”)`도 범례를 표시하지 않으며 `theme(legend.position=“top”)`는 범례의 위치를 위쪽으로 이동시켜준다.
```r
# case 1 : guides(fill=FALSE)
ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) + geom_bar(stat = "identity") + guides(fill=FALSE)
# case 2 : scale_fill_discrete(guide=FALSE)
ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) + geom_bar(stat = "identity") + scale_fill_discrete(guide=FALSE)
# case 3 : theme(legend.position="top")
ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) + geom_bar(stat = "identity") + theme(legend.position="top")
```

<center><img src="/assets/Basic_lecture7/12.png" width="500" height="400"></center>

<center><img src="/assets/Basic_lecture7/13.png" width="500" height="400"></center>

<center><img src="/assets/Basic_lecture7/14.png" width="500" height="400"></center>

또한 y축에 변수 1개의 정보만을 담기보다 여러 변수의 정보를 담아야 할때가 많다. 

이 때 `melt`함수를 이용해준다. 
```r
library(reshape)
new_iris <- melt(iris, id.vars = "Species")
head(new_iris)
dim(new_iris)
```

`melt`는 4개의 변수를 1개의 변수로 만들어주는 역할을 한다. 
```r
ggplot(new_iris, aes(x = Species, y = value, fill = variable)) + geom_bar(stat = "identity")
```

<center><img src="/assets/Basic_lecture7/15.png" width="500" height="400"></center>

또한 변수가 합쳐진 그래프가 아니라 따로 떨어진 형태로 만들고 싶을 땐 `position=“dodge”`를 사용한다.
```r
ggplot(new_iris, aes(Species, value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge")
```

<center><img src="/assets/Basic_lecture7/16.png" width="500" height="400"></center>

### 2-3-3. Histogram

- 히스토그램은 도수분포표를 자료로 하여, 계급구간을 밑변으로 하고 도수를 높이로 하여 그린 그림

- 기둥 간에 간격이 없는 것이 Bar plots와 다름

- `binwidth`로 막대의 넓이는 조절

- `fill`은 내부 채우는 색을 의미하고, `colour`은 테두리를 의미

```r
h <- ggplot(iris, aes(x = Sepal.Width))
# case 1
h + geom_histogram(color = "black", fill = "steelblue")
# case 2
h + geom_histogram(binwidth= 0.5, color = "white", fill = "tomato")
```

<center><img src="/assets/Basic_lecture7/17.png" width="500" height="400"></center>

<center><img src="/assets/Basic_lecture7/18.png" width="500" height="400"></center>

### 2-3-3. Density plot

Density curve를 그리기 위해서는 `geom_density()`를 이용한다.
```r
ggplot(iris, aes(x = Sepal.Width)) + geom_density()
```

<center><img src="/assets/Basic_lecture7/19.png" width="500" height="400"></center>

`fill` 옵션을 통해 내부를 색으로 채우고, `alpha` 옵션은 투명도를 나타낸다.
```r
ggplot(iris, aes(x = Sepal.Width)) + geom_density(fill = "blue", alpha = 0.5)
```

<center><img src="/assets/Basic_lecture7/20.png" width="500" height="400"></center>

위의 Histogram과 Density plot를 함께 그린 시각화는 아래와 같다.
```r
h + geom_histogram(binwidth=0.2, color = "black", fill = "tomato", aes(y=..density..)) + geom_density(fill = "blue", alpha = 0.3)
```

<center><img src="/assets/Basic_lecture7/21.png" width="500" height="400"></center>

축의 이름을 설정하는 것은 다음과 같다.
```r
# case 1
h + geom_histogram(binwidth=0.2, colour = "black", fill = "tomato", aes(y=..density..)) + geom_density(fill = "blue", alpha = 0.3) + xlab("Sepal Width") +  ylab("Density") + ggtitle("Histogram & Density Curve")
# case 2
h + geom_histogram(binwidth=0.2, colour = "black", fill = "tomato", aes(y=..density..)) + geom_density(fill = "blue", alpha = 0.3) + labs(x = "Sepal Width", y = "Density", title = "Histogram & Density Curve")
```

<center><img src="/assets/Basic_lecture7/22.png" width="500" height="400"></center>

<center><img src="/assets/Basic_lecture7/23.png" width="500" height="400"></center>

### 2-3-4. 3차원 그래프

Lecture6에서 보았던 3차원 그래프는 그래프의 점을 확인하는 데 있어서 자유도가 낮았다.

따라서 `plotly`를 이용해 자유도가 높은 그래프를 그릴 수 있으며 문법은 `ggplot2`와 매우 유사하다.
```r
library(plotly)
plot_ly(data = iris, x = ~Petal.Length, y = ~Petal.Width, z = ~Sepal.Length, type = "scatter3d", mode = "markers", color = ~Species)
```

<center><img src="/assets/Basic_lecture7/24.png" width="500" height="400"></center>

### 2-3-5. 좀 더 interactive하게!

`plotly`는 isnteractive 그래프를 그려주는 라이브러리이다.

`plotly` 라이브러리의 `ggplotly`를 이용하여 interactive한 그래프를 만들 수 있다.
```r
p <- ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) + geom_point(size = 2)
ggplotly(p)
```

<center><img src="/assets/Basic_lecture7/25.png" width="500" height="400"></center>

```r
q <- ggplot(iris, aes(x = Sepal.Width)) + geom_histogram(colour = "white", fill = "tomato")
ggplotly(q)
```

<center><img src="/assets/Basic_lecture7/26.png" width="500" height="400"></center>

# 3. Question

`ggplotly`를 이용해 다음의 그래프를 만드시오.

## 3-1. Question

<center><img src="/assets/Basic_lecture7/27.png" width="500" height="400"></center>

## 3-2. Question

<center><img src="/assets/Basic_lecture7/28.png" width="500" height="400"></center>
