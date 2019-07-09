---
layout: post
title:  "Multivariate & Data mining report"
date:   2019-05-24
use_math: true
tags:
 - R
 - korean
 - Report
---

# 관측값이 적은 상황에서의 대표모델 선택에 관한 연구

## 1. Introduction

지도학습(Supervised learning)에서는 풍부한 훈련데이터가 요구된다. 
그러나 풍부한 훈련데이터가 없을 때 모집단을 대표할 수 있는 모델링에 대한 선택이 필요할 경우가 종종 있다. 
따라서 본 논문에서는 적은 훈련데이터로 학습을 시켰을 때, 가장 적은 오차를 갖는 모델링이 무엇인지 알아보고자 한다. 
이에 대한 방법은 Cross-Validation(CV)을 통해 검증을 할 것인데, 보통의 CV는 훈련데이터와 예측데이터의 비율을 8:2로 정하지만 여기에서는 2:8의 비율을 통해 검증하고자 한다. 
하지만 적은 비율을 갖는 훈련데이터의 관측값이 많아서는 안되기 때문에, 전체 관측값의 크기를 200으로 잡고 훈련데이터의 크기를 40개로 제한하도록 한다. 
2절에서는 사용할 모델링에 대한 소개를 진행하고, 3절에서는 시각화를 위한 변수의 갯수가 1개일 경우를 시뮬레이션하고 4절에서는 일반적인 상황에서도 사용할 수 있도록 검증하는 단계를 거쳐 5절에서 정리하며 마치도록 하겠다.

## 2. Models

이 절에서는 앞으로 사용할 지도학습 모델링에 대해서 간략하게 설명한다.

### 2-1. Linear regression

선형모형이란 F.Galton(1885)에 의해 개발되었으며 출력변수를 입력변수들의 선형결합으로 표현한 모델이며 이를 수식으로 나타내면 다음과 같다.

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \epsilon \quad \epsilon \sim N(0, \sigma^2)
$$

이며 확률오차는 정규분포이며 등분산과 독립성을 가정한다. 
입력변수와 출력변수의 평균과의 관계를 설명하는 식을 찾는 것이 목적이며 데이터에 대한 초기탐색을 할 때에 주로 사용한다. 

### 2-2. Logistic regression

로지스틱 회귀모형이란 DR.Cox(1958)에 의해 개발되었으며 선형모형과 유사하지만 확률의 발생가능성을 예측한다는 점에서 차이가 있다. 
발생가능성을 예측한다는 점에서 출력변수가 비선형결합으로 이루어져 있는데 이를 수식으로 나타내면 다음과 같다.

$$
\text{logit}(p) = \log\frac{p}{1 - p} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p \quad P(Y = 1|X) = p = \frac{1}{1 + \exp(-X\beta)}
$$

위의 식을 이용해 발생확률의 기준을 0.5 혹은 다른 확률로 잡고 이보다 크면 1, 작으면 0으로 두고 분류한다.

### 2-3. Random forest

다수의 의사결정나무를 학습하는 앙상블(Ensemble)기법으로 Leo.Breiman(2001)에 의해 개발되었다.
의사결정나무 모델은 과적합 문제가 항상 존재한다. 
이를 해결하고자 랜덤포레스트는 서로 다른 특징을 갖는 의사결정나무를 생성해 일반적인 성능을 향상시키는데 목적이 있다. 
또한 회귀분석문제와 분류문제를 동시에 해결할 수 있으며 부스트랩을 통한 일반화로 좋은 성능을 기대할 수 있다고 알려져 있다. 
사용할 R 라이브러리는 `randomForest`을 사용하겠다.

### 2-4. Support vector machine

p개의 데이터를 분류하는 데에 가장 쉬운 방법 중 하나는 (p - 1)차원의 초평면으로 어디에 속할지 나누는 것이다. 
이러한 아이디어를 바탕으로 Vapnik(1995)가 개발했으며 초평면을 기준으로 위에 있는지, 아래에 있는지를 분류하고 이 분류하는 거리를 계산해 최대화하는 방법이다. 
이를 수식으로 나타내면 다음과 같다.

$$
\text{minimize}\frac{1}{2}||w||^2 + C\sum_{i = 1}^n \max (1 - y_i(w^tg(x_i) + b), 0) \quad C \:\text{ is margin parameter}
$$

으로 나타내진다. 여기에서 $g(\cdot)$은 비선형 결합으로, 선형으로 분류 혹은 예측할 수 없는 현실의 문제에서 kernel이라는 비선형 결합을 이용해 비선형 분류 혹은 비선형 회귀를 가능하게 한다. 
사용할 R 라이브러리는 `e1071`을 사용하겠다.

### 2-5. Neural network

인공신경망은 W.McCulloch, W.Pitts(1943)에 의해 신경망기법이 개발되었으며 F.Rosenblatt(1958)의 퍼셉트론, B.Widrow, T.Hoff(1960)의 Adaline으로 계속 개발되어왔다. 
Support vector machine과 유사한 점이 굉장히 많으며 여러개의 층을 쌓아서 분류 혹은 회귀를 할 있다는 점에서 장점이 있다.
이 때 사용하는 비용함수는 다음과 같다.

$$
\begin{split}
&\text{minimize}\sum_{i = 1}^n(y_i - g(w^tx_i))^2 \quad \text{in regression, classification}\\
&\text{minimize}\sum_{i = 1}^ny_i \log(g(w^tx_i)) \quad \text{in classification}
\end{split}
$$

중요한 것은 여기에서 가중치 $w$는 역전파(Backpropagation)에 의해 업데이트가 이루어지며 업데이트 과정에서 비용함수의 정보가 들어가기 때문에 역전파로 일컬어진다.
사용할 R 라이브러리는 `neuralnet`을 사용하겠다.

### 2-6. K-fold cross validation

주어진 데이터를 훈련데이터와 예측데이터로 만들어진 모델에 대한 성능평가를 실행할 때 많이 쓰이는 기법이다. 데이터의 일부분만 검증에 사용되며 훈련데이터와 예측데이터를 나누는 최적의 비율의 이론적인 근거는 존재하지 않지만 7:3을 많이 사용한다. 또한 과적합을 방지할 수 있으며 최적의 모수를 선택하거나 변수를 선택하는 데에 쓰이기도 한다. 다음의 과정으로 k-fold CV가 진행된다.

- **step1)** 데이터를 랜덤하게 K개의 같은 크기로 분할한다.
- **step2)** (K-1)-fold의 데이터를 훈련데이터로 사용해 원하는 모델을 적합시킨다.
- **step3)** 나머지 1-fold의 데이터(fold-out data)를 이용해 출력변수를 예측한다.
- **step4)** step1 ~ step3의 과정을 K번 반복하여 예측한다.
- **step5)** step1 ~ step4의 과정을 각각의 후보 모형마다 실시한다. 
- **step6)** 최적의 모델을 선택한다.

위에서 언급한대로 훈련데이터를 7, 예측데이터를 3의 비율로 사용하지만 우리의 목적은 적은 관측치에 대해서 어떤 모델을 선택할 것인가에 관한 문제이기 때문에 훈련데이터를 2, 예측데이터를 8의 비율로 사용해 좋은 모형을 결정한다. 

## 3. Simple simulation

이 절에서는 시각화를 위한 1차원 데이터를 사용한다.

### 3-1. Regression

이 분석에서 사용할 모델의 옵션은 다음과 같다.

- Linear regression : QR decomposition
- Random forest : ntree = 500, Entropy error
- Support vector regression : Gaussian kernel, epsilon = 0.1, gamma = 1
- Neural network : threshold = 0.1, hidden = c(2, 2)

#### 3-1-1. Linear regression

시뮬레이션에 사용할 독립변수 $X$는 $U(0, 2)$에서 추출하며 종속변수는

$$
Y_i = X_i -2 + \epsilon_i, \quad i = 1, 2, \ldots, 200
$$

이고 $\epsilon_i$는 평균이 0이고 표준편차가 $\sqrt{X_i}$인 정규분포로부터 추출한다. 데이터의 분포는 다음과 같다.

<center><img src="/assets/Multivariate & Data mining - report/1.png"></center>

시각화를 위해 적합결과를 그래프로 나타내 본다.

<center><img src="/assets/Multivariate & Data mining - report/2.png"></center>

|  || **Linear regression** || **Random forest** | |**Support vector regression** || **Neural network** | 
|:---|-|:---:|-|:---:|-|:---:|-|:---:|
|MSE|| 0.9633929 || 1.162478| | 1.053037| | 1.002641 |

결과는 위와 같으며 선형회귀분석은 직선의 선으로만 이어진 것과 달리 SVR과 Neural network는 곡선의 형태로 이루어져 있다. 주목할 것은 Random forest인데, 학습데이터 셋의 관측치의 갯수가 20개이므로 과적합되어있음을 알 수 있다. MSE관점에서도 제일 높은 것으로 미루어보아 좋은 모델링이 아니라는 것을 알 수 있다. 그러나 에러를 바꿔주면서 시뮬레이션을 통해 확인을 해야하므로 100번과 400번의 시뮬레이션을 진행한다.

|**Model**|| MSE(100 times simulation) || MSE(400 times simulation) |
|:---|-|:---:|-|:---:|
|**Linear regression**| |1.046357 || 1.053194 |
|**Random forest** || 1.341278 || 1.352666 |
|**Support vector regression**|| 1.124858| | 1.130257 |
|**Neural network**| |1.116376 || 1.124572 |

네개의 모델링 전부 MSE가 100번과 400번의 차이가 0.01정도 밖에 나지 않기 때문에 분산이 작은 것을 알 수 있다. 
또한 위에서 예상한 바와 같이 Random forest는 과적합의 문제가 발생한다는 것 또한 알 수 있다.

#### 3-1-2. Non-linear regression

시뮬레이션에 사용할 독립변수 $X$는 $U(0, 2)$에서 추출하며 종속변수는

$$
Y_i = -X_i^3 + 2X_i - 5 + \epsilon_i, \quad i = 1, 2, \ldots, 200
$$

이고 $\epsilon_i$는 평균이 0이고 표준편차가 $\sqrt{X_i}$인 정규분포로부터 추출한다. 데이터의 분포는 다음과 같다.

<center><img src="/assets/Multivariate & Data mining - report/3.png"></center>

시각화를 위해 적합결과를 그래프로 나타내 본다.

<center><img src="/assets/Multivariate & Data mining - report/4.png"></center>

|  || **Linear regression** || **Random forest** || **Support vector regression** || **Neural network** | 
|:---|-|:---:|-|:---:|-|:---:|-|:---:|
|MSE|| 2.021668 || 1.278364 || 1.219632 || 1.016379|

결과는 위와 같으며 선형회귀분석은 직선의 선으로만 이어진 것과 달리 SVR과 Neural network는 곡선의 형태로 이루어져 있다. 주목할 것은 Random forest인데, 학습데이터 셋의 관측치의 갯수가 20개이므로 과적합되어있음을 알 수 있다. MSE관점에서는 선형으로만 적합되는 선형회귀분석이 가장 좋지 않음을 알 수 있다. 그러나 에러를 바꿔주면서 시뮬레이션을 통해 확인을 해야하므로 100번과 400번의 시뮬레이션을 진행한다.

|**Model**|| MSE(100 times simulation) || MSE(400 times simulation) |
|:---|-|:---:|-|:---:|
|**Linear regression**| |1.984245 | |1.957363 |
|**Random forest**|| 1.371279 | |1.378264 |
|**Support vector regression**|| 1.234569 || 1.244264 |
|**Neural network**|| 1.307873 || 1.345574 |

이 표로 미루어 보았을 때, SVR의 성능이 제일 좋음을 알 수 있고 성능이 제일 좋을 것이라 기대했던 Neural network는 SVR보다 낮은 성능을 보이고 100번과 400번의 시뮬레이션의 분산이 큰 것으로 보아 모델 적합에 어려움을 겪을 수 있다고 볼 수 있다.

### 3-2. Classification

이 분석에서 사용할 모델의 옵션은 다음과 같다.

- Logistic regression : logit function
- Random forest : ntree = 500, Entropy error
- Support vector macine : Gaussian kernel, epsilon = 0.1, gamma = 1
- Neural network : threshold = 0.1, hidden = c(4, 4)

#### 3-2-1. Linearly seperable

시뮬레이션에 사용할 독립변수 $X_{200\times 2}$는 $N(0, 1)$에서 추출하며 종속변수는

$$
Y_i = sign(0.1 + 2X_i + \epsilon_i), \quad i = 1, 2, \ldots 200
$$

이고 $\epsilon_i$는 평균이 0이고 표준편차가 1인 정규분포로부터 추출한다. 

<center><img src="/assets/Multivariate & Data mining - report/5.png"></center>

시각화를 위해 적합결과를 그래프로 나타내 본다.

<center><img src="/assets/Multivariate & Data mining - report/6.png"></center>

|  | |**Logistic regression** |  |**Random forest** | | **Support vector machine** | | **Neural network** | 
|:---|-|:---:|- |:---:|- |:---:|- |:---:|
|Accuracy| | 0.85 | | 0.7875 | | 0.79375 | | 0.8 |

결과는 위와 같으며 선형으로 분류할 수 있기 때문에 어떤 모델에서도 성능을 보장받을 수 있을 것으로 기대한다. 그러나 Random forest의 경우 학습데이터 셋의 관측치의 갯수가 20개이므로 과적합되어있음을 알 수 있다. 실제 예측 정확도에서도 제일 낮기 때문에 좋은 모델링이 아니라는 것을 알 수 있다. 그러나 에러를 바꿔주면서 시뮬레이션을 통해 확인을 해야하므로 100번과 400번의 시뮬레이션을 진행한다.

|**Model**| | Accuracy(100 times simulation) | | Accuracy(400 times simulation) |
|:---|- |:---:|- |:---:|
|**Logistic regression**| | 0.8814625 | | 0.8784312 |
|**Random forest**| |0.8339 | | 0.8323875 |
|**Support vector machine**| | 0.8628125 | | 0.8610813 |
|**Neural network**| | 0.8770375 | | 0.8734312 |

과적합으로 예상했던 Random forest의 예측정확도가 제일 낮기 때문에 좋은 모델링은 아니라는 것을 알 수 있다. 

#### 3-2-2. Non-linearly seperable

시뮬레이션에 사용할 독립변수 $X_{200\times 2}$는 $N(0, 1)$에서 추출하며 종속변수는

$$
Y_i = sign(\log(X_{i1}^2 + X_{i2}^2)+ \epsilon_i), \quad i = 1, 2, \ldots 200
$$

이고 $\epsilon_i$는 평균이 0이고 표준편차가 1인 정규분포로부터 추출한다. 

<center><img src="/assets/Multivariate & Data mining - report/7.png"></center>

시각화를 위해 적합결과를 그래프로 나타내 본다.

<center><img src="/assets/Multivariate & Data mining - report/8.png"></center>

|  | |**Logistic regression** || **Random forest** || **Support vector machine** || **Neural network** | 
|:---|-|:---:|-|:---:|-|:---:|-|:---:|
|Accuracy|| 0.4675 || 0.75625 || 0.78125 || 0.74375 |

결과는 위와 같으며, 비선형으로 이루어진 분류문제이기 때문에 로지스틱회귀는 50% 정도의 성능만을 기대할 수 있다. 
눈에 띄는 부분은 SVM의 성능이 다른 분류 모델들과 2%의 성능 차이를 갖고 있다. 
그러나 에러를 바꿔주면서 시뮬레이션을 통해 확인을 해야하므로 100번과 400번의 시뮬레이션을 진행한다.

|**Model**| |Accuracy(100 times simulation) || Accuracy(400 times simulation) |
|:---|-|:---:|-|:---:|
|**Logistic regression**|| 0.4949875 || 0.4995812 |
|**Random forest**|| 0.710425 || 0.7095375 |
|**Support vector machine**|| 0.739725 || 0.73875 |
|**Neural network**|| 0.6416250 || 0.6402344 |

결과는 위의 표와 같고 Random forest와 Neural network가 과적합의 문제가 있을 것으로 기대했기 때문에 성능 또한 보장받지 못했다.

## 4. Generalization

이번 절에서는 일반화를 위한 검증을 실시한다. 이를 위해 20가지 상황을 가정하며 사용할 모델의 옵션은 아래와 같다.

- Linear regression : QR decomposition
- Logistic regression : logit function
- Random forest : ntree = 500, Entropy error
- Support vector regression : Gaussian kernel, epsilon = 0.1, gamma = 1
- Support vector macine : Gaussian kernel, epsilon = 0.1, gamma = 1
- Neural network(regression) : threshold = 10, hidden = c(100, 100)
- Neural network(classification) : threshold = 1, hidden = c(100, 100)

### 4-1. Regression

| $Y$ || $X$ || $\beta$ || $\epsilon$ |
|:-------------------------:|-|:----------------------------:|-|:---:|-|:---:|
| $X\beta + \epsilon$ || $N(0, 1)$ || $1$ || $N(0, 1)$ |
| $X\beta + \epsilon$ || $N(0, 1)$ || $1$ || $\chi^2_1$ |
| $X\beta + \epsilon$ || $V_{ij} = Cov(X)_{ij} = 0.9^{abs(i - j)}$ || $1$ || $N(0, 1)$ |
| $X\beta + \epsilon$ || $V_{ij} = Cov(X)_{ij} = 0.9^{abs(i - j)}$ || $1$ || $\chi^2_1$ |
| $\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_{50} x_{50} + \epsilon$ || $MVN(\mu, I), \quad \mu_j \sim \chi^2_j$ || $-1$ || $N(0, 1)$ |
| $\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_{50} x_{50} + \epsilon$ || $MVN(\mu, I), \quad \mu_j \sim \chi^2_j$ || $-1$ || $\chi^2_1$ |
| $\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_{50} x_{50} + \epsilon$ || $MVN(\mu, V), \quad \mu_j \sim \chi_j^2$ || $-1$ || $N(0, 1)$ |
| $\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_{50} x_{50} + \epsilon$ || $MVN(\mu, V), \quad \mu_j \sim \chi_j^2$ || $-1$ || $\chi^2_1$ |
| $\beta_0 + \beta_1 \exp(x) + \cdots + \beta_{50} \exp(x) + \epsilon$ || $N(1, 2)$ || $(-1)^{j}$ || $N(0, 1)$ |
| $\beta_0 + \beta_1 \exp(x) + \cdots + \beta_{50} \exp(x) + \epsilon$ || $N(1, 2)$ || $(-1)^{j}$ || $\chi^2_2$ |
| $\beta_0 + \beta_1 \exp(x) + \cdots + \beta_{50} \exp(x) + \epsilon$ || $V_{ij} = Cov(X)_{ij} = 0.9^{abs(i - j)}$ || $(-1)^{j}$ || $N(0, 1)$ |
| $\beta_0 + \beta_1 \exp(x) + \cdots + \beta_{50} \exp(x) + \epsilon$ || $V_{ij} = Cov(X)_{ij} = 0.9^{abs(i - j)}$ || $(-1)^{j}$ || $\chi^2_2$ |
| $\beta_0 + \beta_1 \cos(x) + \cdots + \beta_{50} \sin(x) + \epsilon$ || $N(0, 1)$ || $(-1)^{j}$ || $N(0, 1)$ ||
| $\beta_0 + \beta_1 \cos(x) + \cdots + \beta_{50} \sin(x) + \epsilon$ || $N(0, 1)$ || $(-1)^{j}$ || $\chi^2_1$ ||
| $\beta_0 + \beta_1 \cos(x) + \cdots + \beta_{50} \sin(x) + \epsilon$ || $V_{ij} = Cov(X)_{ij} = 0.9^{abs(i - j)}$ || $(-1)^{j}$ || $N(0, 1)$ |
| $\beta_0 + \beta_1 \cos(x) + \cdots + \beta_{50} \sin(x) + \epsilon$ || $V_{ij} = Cov(X)_{ij} = 0.9^{abs(i - j)}$ || $(-1)^{j}$ || $\chi^2_1$ |
| $X\beta + \epsilon$ || $MVN(\mu, I), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $N(0, 1)$ |
| $X\beta + \epsilon$ || $MVN(\mu, I), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $\chi^2_2$ |
| $X\beta + \epsilon$ || $MVN(\mu, V), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $N(0, 1)$ |
| $X\beta + \epsilon$ || $MVN(\mu, V), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $\chi^2_2$ |

|**Case**|| **Linear regression** || **Random forest** || **Support vector regression** || **Neural network** |
|:---|-|:-----------:|-|:-----------:|-|:----------------------:|-|:-----------:|
|case1||628063(4) || 44.74(2) || 41.03(1) || 54.27(3) |
|case2||127475(4) || 45.67(2) || 41.92(1) || 54.75(3) |
|case3||657466(4) || 202.06(3)|| 188.79(2)|| 115.85(1)|
|case4||8868125(4)|| 205.27(3)|| 190.13(2)|| 118.90(1)|
|case5||728069508(4)||2202(2)||1970(1)||2606(3)|
|case6||36479166(4)||2190(2)||1957(1)||2590(3)|
|case7||681602(4)||200.53(2)||191.57(1)||785.03(3)|
|case8||735166(4)||201.25(2)||191.08(1)||792.23(3)|
|case9||26486065(4)||3844.59(3)||3689.349(1)||3765.20(2)|
|case10||1112395655(4)||17312(2)||16674(1)||18133(3)|
|case11||1036249(4)||182.65(2)||175.64(1)||186.62(3)|
|case12||9862885(4)||170.80(2)||167.44(1)||176.74(3)|
|case13||218244(4)||14.45(2)||12.96(1)||20.41(3)|
|case14||31881(4)||15.62(2)||14.11(1)||21.89(3)|
|case15||122203(4)||24.07(2)||17.73(1)||38.31(3)|
|case16||4950470853(4)||24.74(2)||18.59(1)||38.61(3)|
|case17||230566(4)||44.62(2)||41.09(1)||54.78(3)|
|case18||52377(4)||48.07(2)||44.35(1)||57.93(3)|
|case19||1834452(4)||200.32(2)||191.18(1)||806.56(3)|
|case20||594589(4)||201.51(2)||188.25(1)||793.29(3)|

### 4-2. Classification

| $Y$ || $X$ || $\beta$ || $\epsilon$ |
|:---------------------------:|-|:--------------------------:|-|:---:|-|:---:|
| $sign(X\beta + \epsilon - \bar{Y})$ || $N(0, 1)$ || $1$ || $N(0, 1)$ |
| $sign(X\beta + \epsilon - \bar{Y})$ || $N(0, 1)$ || $1$ || $\chi^2_1$ |
| $sign(X\beta + \epsilon - \bar{Y})$ || $V_{ij} = Cov(X)_{ij} = 0.9^{ij}$ || $1$ || $N(0, 1)$ |
| $sign(X\beta + \epsilon - \bar{Y})$ || $V_{ij} = Cov(X)_{ij} = 0.9^{ij}$ || $1$ || $\chi^2_1$ |
| $sign(\beta_0 + \beta_1 x_1 + \cdots + \beta_{50} x_{50} + \epsilon - \bar{Y})$ || $MVN(\mu, I), \quad \mu_j \sim \chi^2_j$ || $-1$ || $N(0, 1)$ |
| $sign(\beta_0 + \beta_1 x_1 + \cdots + \beta_{50} x_{50} + \epsilon - \bar{Y})$ || $MVN(\mu, I), \quad \mu_j \sim \chi^2_j$ || $-1$ || $\chi^2_1$ |
| $sign(\beta_0 + \beta_1 x_1 + \cdots + \beta_{50} x_{50} + \epsilon - \bar{Y})$ || $MVN(\mu, V), \quad \mu_j \sim \chi_j^2$ || $-1$ || $N(0, 1)$ |
| $sign(\beta_0 + \beta_1 x_1 + \cdots + \beta_{50} x_{50} + \epsilon - \bar{Y})$ || $MVN(\mu, V), \quad \mu_j \sim \chi_j^2$ || $-1$ || $\chi^2_1$ |
| $sign(\beta_0 + \cdots + \beta_{50} \exp(x) + \epsilon -\bar{Y})$ || $N(1, 2)$ || $(-1)^{j}$ || $N(0, 1)$ |
| $sign(\beta_0 + \cdots + \beta_{50} \exp(x) + \epsilon -\bar{Y})$ || $N(1, 2)$ || $(-1)^{j}$ || $\chi^2_2$ |
| $sign(\beta_0 + \cdots + \beta_{50} \exp(x) + \epsilon -\bar{Y})$ || $V_{ij} = Cov(X)_{ij} = 0.9^{ij}$ || $(-1)^{j}$ || $N(0, 1)$ |
| $sign(\beta_0 + \cdots + \beta_{50} \exp(x) + \epsilon -\bar{Y})$ || $V_{ij} = Cov(X)_{ij} = 0.9^{ij}$ || $(-1)^{j}$ || $\chi^2_2$ |
| $sign(\beta_0 + \cdots + \beta_{50} \sin(x) + \epsilon -\bar{Y})$ || $N(0, 1)$ || $(-1)^{j}$ || $N(0, 1)$ |
| $sign(\beta_0 + \cdots + \beta_{50} \sin(x) + \epsilon -\bar{Y})$ || $N(0, 1)$ || $(-1)^{j}$ || $\chi^2_1$ |
| $sign(\beta_0 + \cdots + \beta_{50} \sin(x) + \epsilon -\bar{Y})$ || $V_{ij} = Cov(X)_{ij} = 0.9^{ij}$ || $(-1)^{j}$ || $N(0, 1)$ |
| $sign(\beta_0 + \cdots + \beta_{50} \sin(x) + \epsilon -\bar{Y})$ || $V_{ij} = Cov(X)_{ij} = 0.9^{ij}$ || $(-1)^{j}$ || $\chi^2_1$ |
| $sign(X\beta + \epsilon -\bar{Y})$ || $MVN(\mu, I), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $N(0, 1)$ |
| $sign(X\beta + \epsilon -\bar{Y})$ || $MVN(\mu, I), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $\chi^2_2$ |
| $sign(X\beta + \epsilon -\bar{Y})$ || $MVN(\mu, V), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $N(0, 1)$ |
| $sign(X\beta + \epsilon -\bar{Y})$ || $MVN(\mu, V), \quad \mu_j \sim U((-1)^j, 2j)$ || $1$ || $\chi^2_2$ |

|**Case**|| **Logistic regression** || **Random forest** || **Support vector machine** || **Neural network** |
|:---|-|:------------:|-|:------------:|-|:-------------------:|-|:------------:|
|case1||0.533(3)||0.614(2)||0.621(1)||0.499(4)|
|case2||0.530(4)||0.618(2)||0.626(1)||0.505(3)|
|case3||0.549(4)||0.863(2)||0.912(1)||0.565(3)|
|case4||0.549(3)||0.861(2)||0.911(1)||0.545(4)|
|case5||0.519(4)||0.611(2)||0.630(1)||0.498(4)|
|case6||0.524(3)||0.610(2)||0.624(1)||0.496(4)|
|case7||0.543(3)||0.863(2)||0.912(1)||0.497(4)|
|case8||0.549(3)||0.866(2)||0.913(1)||0.500(4)|
|case9||0.537(3)||0.580(2)||0.600(1)||0.525(4)|
|case10||0.527(4)||0.558(2)||0.570(1)||0.547(3)|
|case11||0.520(3)||0.528(1)||0.527(2)||0.503(4)|
|case12||0.527(4)||0.571(2)||0.585(1)||0.546(3)|
|case13||0.533(3)||0.612(2)||0.625(1)||0.501(4)|
|case14||0.534(3)||0.623(2)||0.636(1)||0.513(4)|
|case15||0.546(3)||0.816(1)||0.807(2)||0.506(4)|
|case16||0.541(3)||0.815(1)||0.810(2)||0.512(4)|
|case17||0.533(3)||0.611(2)||0.616(1)||0.499(4)|
|case18||0.531(3)||0.635(1)||0.634(2)||0.521(4)|
|case19||0.549(3)||0.864(2)||0.911(1)||0.497(4)|
|case20||0.549(3)||0.861(2)||0.906(1)||0.499(4)|

## 5. Conclusion

위의 과정을 통해 회귀분석과 분류분석 시뮬레이션을 실시하였다. 풍부한 학습데이터를 필요로하는 지도학습에서 부족한 학습데이터에 대해서도 모집단을 대표할 수 있는 방법론을 찾는 것이 목적이었다. 4장에서 보이는 표에서도 쉽게 알 수 있듯이, SVR, SVM이 다른 방법론들에 비해 좋은 결과를 갖는다는 것을 알게 되었다. 이 뿐만이 아니라, 보통 분석을 하는 과정은 EDA를 실시한 후에 초기 모델을 설정하고 Hyperparameter를 계속해서 수정해, 여러 방법론을 비교하는 것이 일반적이다. Hyperparameter를 수정할 때 Cross-Validation(CV)로 수정을 해나가는 것은 시간이 오래걸리기 마련이다. 그러나 초기 모델을 SVM으로 설정하고 Hyperparameter를 수정해나가면 다른 방법론을 비교하지 않아도 초기 모델 SVM이 가장 좋은 결과를 가져올 수 있다는 것을 기대할 수 있다. 따라서 시간을 절약할 수 있다는 점에서 본 논문은 의미가 있다고 할 수 있다.

## Reference

Gareth James, Daniela Witten, Trevor Hastie and Rob Tibshirani (2017). **ISLR**: *Data for an Introduction to Statistical Learning with Applications in R*. R package version 1.2. https://CRAN.R-project.org/package=ISLR

Fritsch, Stefan, Frauke Guenther, and Marvin N. Wright. 2019. Neuralnet: Training of Neural Networks. https://CRAN.R-project.org/package=neuralnet.

w, Andy, and Matthew Wiener. 2002. “Classification and Regression by randomForest.” R News 2 (3):1822. https://CRAN.R-project.org/doc/Rnews/.

Meyer, David, Evgenia Dimitriadou, Kurt Hornik, Andreas Weingessel, and Friedrich Leisch. 2019. E1071:Misc Functions of the Department of Statistics, Probability Theory Group (Formerly: E1071), Tu Wien. https://CRAN.R-project.org/package=e1071.
