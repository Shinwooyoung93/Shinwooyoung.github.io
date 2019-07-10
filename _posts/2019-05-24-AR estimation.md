---
layout: post
title:  "Statistical Signal Processing report"
date:   2019-05-24
use_math: true
tags:
 - R
 - english
 - Report
---

# AR(1) estimation with quantile loss

## Abstract

The Autoregressive(AR) model is used to describe certain time-varying processes in nature, economics, etc. 
There are three parameters to be estimated in this AR model. 
There are three ways to estimate this, including the MLE, LSE, and Yule-walker methods taught in class. 
We will briefly introduce these methods and look at the differences in the estimated parameters between methods. 
Also we will show two surprising points, the first of which will show through simulation how powerful the sample mean is. 
Secondary, in addition to above estimation method, we try to estimate by using the check loss used for quantile regression.

## 1. AR model

The AR model is a normal probability model consisting of the linear weighted sum of the present value of white noise and its own past value.
AR(p) model using below equation

$$
X_t = \rho_1 X_{t - 1} + \rho_2 X_{t - 2} + \cdots + \rho_p X_{t - p} + e_t, \quad e_t \sim N(0, \sigma^2)
$$

![Figure 1. Example of AR(1)](/assets/Statistical Signal Processing/0.png)

Also it need some assumptions

- $E(X_t)$ is independent of $t$. In other words, $E(X_t) = \mu \quad \forall t$

- $Var(X_t)$ exists and is independent of $t$. In other words, $Var(X_t) = \sigma^2 < \infty$

- $Cov(X_i, X_j)$ is depend on $t$. In other words, $Cov(X_{i + p}, X_{j + p}) = Cov(X_i, X_j)$

In this model, it exists some parameters as $\theta = (\mu, \rho, \sigma^2)^t$. 
In next chapter, we see the estimation methods like MLE, LSE, and Yule-Walker estimations.

## 2. Materials

### 2-1. MLE method

AR(p) model depend on previous times like $f(X_0, \ldots, X_i) = f(X_0)f(X_1|X_0)\cdots f(X_n|X_{n - 1}, \ldots, X_0)$ 
then, AR(1) model only depend on previous time point. 

Thus AR(1) model's likelihood function is $f(X_0, \ldots, X_i) = f(X_0)f(X_1|X_0)\ldots f(X_n|X_{n - 1})$ 
and $X_t = c + \rho X_{t - 1} + e_t$.

Then we can easily know that $E(X_t) = \mu = c/(1 - \rho)$.

$$
\begin{split}
\ell(\mu, \rho, \sigma^2) &= \log f(X_0)f(X_1|X_0)\cdots f(X_n|X_{n - 1}) \\
&= \log N(\mu, \frac{\sigma^2}{1 - \rho^2})N(c + \rho X_0, \sigma^2)\cdots N(c + \rho X_{n - 1}, \sigma^2) \\
&= -\frac{1}{2}\log(2\pi \sigma^2/1 - \rho^2) -\frac{n}{2}\log(2\pi \sigma^2) -\frac{(X_0 - \mu)^2}{2\sigma^2/(1 - \rho^2)} -\frac{1}{2\sigma^2}\sum_{t = 1}^n(X_t - c - \rho X_{t - 1})^2
\end{split}
$$

But it can not be calculated easily, there is no closed form the exact $mle$. So using $Newton-Rhapson$ method that

$$
\hat{\theta}_{mle} =  \hat{\theta}_{mle} - \hat{H}(\hat{\theta}_{mle})^{-1}\hat{s}(\hat{\theta}) \quad \theta = (\mu, \rho, \sigma^2)^t
$$

### 2-2. LSE method

Least square estimation(LSE) is simple but powerfull method.

LSE set $e_t \sim N(0, \sigma^2)$, only use $minimize$ sum of square errors

$$
S(\rho, \mu) =  \sum_{t = 1}^n(X_t - \mu - \rho (X_{t - 1}-\mu))^2 = \sum_{t = 1}^n(X_t - c - \rho X_{t - 1})^2
$$

thus parameters are calculated easily by partial of score function.

$$
\hat{c} = \hat{\mu}(1 - \hat{\rho}) = \frac{1}{n}\sum_{t = 1}^n(X_t - \hat{\rho}X_{t - 1}) \Leftrightarrow \hat{\mu} = \frac{1}{n}\frac{\sum_{t = 1}^n(X_t - \hat{\rho} X_{t - 1})}{1 - \hat{\rho}}
$$

$$
\hat{\rho} = \frac{\sum_{t = 1}^nX_{t - 1}(X_{t} - \hat{c})}{\sum_{t = 1}^n X_{t-1}^2}
$$

$$
\hat{\sigma} = \sqrt{\frac{1}{n - 1}\sum_{t = 1}^n(X_t - \hat{c} - \hat{\rho} X_{t-1})^2}
$$

Also it can be calculated by iterative method.

### 2-3. Yule-Walker method


Yule-walker method is using moment estimation $\bar{X}$. 

Then set $\gamma_{j} = \rho_1\gamma_{j-1} + \rho_2\gamma_{j-2} + \cdots + \rho_p\gamma_{j - p}$ and

$$
\left[\begin{matrix}
1 & \gamma_1 & \gamma_2 & \cdots & \gamma_{p-1}\\
\gamma_{1} & 1 & \gamma_1 & \cdots & \gamma_{p-2}\\
\gamma_{2} & \gamma_1 & 1 & \cdots & \gamma_{p-3}\\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\gamma_{p-1} & \gamma_{p-2} & \gamma_{p-3} & \cdots & 1
\end{matrix}\right]\left[
\begin{matrix}
\rho_1 \\
\rho_2 \\
\rho_3 \\
\vdots \\
\rho_p
\end{matrix}
\right] = \left[
\begin{matrix}
\gamma_1\\
\gamma_2\\
\gamma_3\\
\vdots \\
\gamma_p
\end{matrix}
\right]
$$

Above equation is called Yule-Walker equation. When we deal with AR(1), 

$$
\begin{split}
\hat{\mu} &= \bar{X}\\
\hat{\gamma} &= \hat{\rho} = \frac{\sum_{t = 1}^n (X_{t - 1} - \bar{X})(X_{t} - \bar{X})}{\sum_{t = 1}^n(X_{t - 1} - \bar{X})^2} \\
\hat{\sigma} &=  \sqrt{\frac{1}{n - 1}\sum_{t = 1}^n(X_t - \bar{X})^2}
\end{split}
$$

## 3. Simulation

In this chapter, I will simulate the estimation in three 3 method with 100 times.

Let true parameters $\mu = 2, \sigma = \sqrt{2}, \rho = 0.2$.

$$
\begin{split}
&X_0 \sim N(\mu, \frac{\sigma^2}{1 - \rho^2}) \\
&X_i = \mu + \rho(X_{i - 1} - \mu) + e_i \quad i = 1, \ldots, n\\
&e_i \sim N(0, \sigma^2)
\end{split}
$$

```r
rm(list = ls())

set.seed(1)
par(mfrow = c(1, 1))
n <- 50
mu <- 2
sigma <- sqrt(2)
rho <- 0.2
x0 <- rnorm(1, mu, sigma/sqrt(1 - rho^2))
xi <- c(x0, rep(NA, n))
for(i in 1:n){
  past <- xi[i]
  past <- mu + rho*(past - mu) + rnorm(1, 0, sigma)
  current <- past
  xi[i + 1] <- current
}
plot(xi, pch = 1, xlab = "time", ylab = "x", main = "AR(1) model")
lines(xi, col = "blue", lwd = 2)
```

![Figure 2. True plot for AR(1)](/assets/Statistical Signal Processing/1.png)

then simulation with changing error term via tensorflow

```r
rm(list = ls())

library(tensorflow)
set.seed(1)
n <- 50
mu <- 2
sigma <- sqrt(2)
rho <- 0.2
x0 <- rnorm(1, mu, sigma/sqrt(1 - rho^2))
xi <- c(x0, rep(NA, n))
for(i in 1:n){
  past <- xi[i]
  past <- mu + rho*(past - mu) + rnorm(1, 0, sigma)
  current <- past
  xi[i + 1] <- current
}
xt <- as.matrix(xi[2:51])
xt1<- as.matrix(xi[1:50])

fit1.function <- function(x0, xt, xt1){
  
  sess = tf$Session()
  X_0 = tf$placeholder(shape = shape(), dtype = tf$float32)
  X_t = tf$placeholder(shape = shape(n, 1), dtype = tf$float32)
  X_t1 = tf$placeholder(shape = shape(n, 1), dtype = tf$float32)
  rho = tf$Variable(0.1)
  sigma = tf$Variable(1.0)
  mu = tf$Variable(2.0)
  c = mu*(1 - rho)
  
  cost1 = 1/2*tf$log(2*pi*sigma^2/(1 - rho^2)) + n/2*tf$log(2*pi*sigma^2) +
    (X_0 - mu)^2/(2*sigma^2/(1 - rho^2)) +  1/(2*sigma^2)*tf$reduce_sum((X_t - c - rho*X_t1)^2)
  train_step = tf$train$AdamOptimizer()$minimize(cost1)
  init = tf$global_variables_initializer()
  sess$run(init)
  criteria <- 1e-8
  past <- Inf
  
  for(i in 1:20000){
    sess$run(train_step,feed_dict=dict(X_0=x0, X_t = xt, X_t1= xt1))
    current <- sess$run(cost1,feed_dict=dict(X_0= x0, X_t = xt, X_t1= xt1))
    if(abs(current - past)< criteria) break
    past <- current
  }
  return(list(mu = sess$run(mu), sigma = sess$run(sigma), rho = sess$run(rho)))
}

###########################
fit2.function <- function(x0, xt, xt1){
  
  sess = tf$Session()
  X_0 = tf$placeholder(shape = shape(), dtype = tf$float32)
  X_t = tf$placeholder(shape = shape(n, 1), dtype = tf$float32)
  X_t1 = tf$placeholder(shape = shape(n, 1), dtype = tf$float32)
  rho = tf$Variable(0.1)
  mu = tf$Variable(2.0)
  c = mu*(1 - rho)
  
  cost2 = tf$reduce_sum((X_t - c - rho*X_t1)^2)
  train_step = tf$train$AdamOptimizer()$minimize(cost2)
  init = tf$global_variables_initializer()
  sess$run(init)
  criteria <- 1e-8
  past <- Inf
  
  for(i in 1:20000){
    sess$run(train_step,feed_dict=dict(X_0=x0, X_t = xt, X_t1= xt1))
    current <- sess$run(cost2,feed_dict=dict(X_0= x0, X_t = xt, X_t1= xt1))
    if(abs(current - past)< criteria) break
    past <- current
  }
  sigma = sqrt(1/(length(xt) - 1)*sum((xt - sess$run(c) - sess$run(rho)*xt1)^2))
  return(list(mu = sess$run(mu), rho = sess$run(rho), sigma = sigma))
}

fit3.function <- function(x0, xt, xt1){
  mu <- mean(xt)
  rho<- sum((xt1 - mean(xt))*(xt - mean(xt)))/sum((xt1 - mean(xt))^2)
  sigma <- sqrt(1/(n - 1)*sum((xt - mu)^2))
  return(list(mu = mu, rho = rho, sigma = sigma))
}

fit1 <- fit1.function(x0, xt, xt1)
fit2 <- fit2.function(x0, xt, xt1)
fit3 <- fit3.function(x0, xt, xt1)

num.simul <- 100
mu.matrix <- matrix(NA, num.simul, 3)
rho.matrix <- matrix(NA, num.simul, 3)
sigma.matrix <- matrix(NA, num.simul, 3)

for(iter in 1:num.simul){
  n <- 50
  mu <- 2
  sigma <- sqrt(2)
  rho <- 0.2
  x0 <- rnorm(1, mu, sigma/sqrt(1 - rho^2))
  xi <- c(x0, rep(NA, n))
  for(i in 1:n){
    past <- xi[i]
    past <- mu + rho*(past - mu) + rnorm(1, 0, sigma)
    current <- past
    xi[i + 1] <- current
  }
  xt <- as.matrix(xi[2:51])
  xt1<- as.matrix(xi[1:50])
  
  fit1 <- fit1.function(x0, xt, xt1)
  fit2 <- fit2.function(x0, xt, xt1)
  fit3 <- fit3.function(x0, xt, xt1)
  
  mu.matrix[iter,1] <- fit1$mu
  rho.matrix[iter,1] <- fit1$rho
  sigma.matrix[iter,1] <- fit1$sigma
  
  mu.matrix[iter,2] <- fit2$mu
  rho.matrix[iter,2] <- fit2$rho
  sigma.matrix[iter,2] <- fit2$sigma
  
  mu.matrix[iter,3] <- fit3$mu
  rho.matrix[iter,3] <- fit3$rho
  sigma.matrix[iter,3] <- fit3$sigma
}
write.csv(mu.matrix, "mu_data.csv", row.names = F)
write.csv(rho.matrix, "rho_data.csv", row.names = F)
write.csv(sigma.matrix, "sigma_data.csv", row.names = F)

rm(list = ls())
mu <- read.csv("mu_data.csv", header = T)
rho <- read.csv("rho_data.csv", header = T)
sigma <- read.csv("sigma_data.csv", header = T)
par(mfrow = c(1, 3))

plot(density(mu[,1]), col = 2, lwd = 2, ylim = c(0, 2), main = "mu estimation")
points(density(mu[,2]), col = 3, type = "l", lwd = 2)
points(density(mu[,3]), col = 4, type = "l", lwd = 2)
legend("topleft", c("MLE", "LSE", "Y-W"), col = 2:4, lwd = 2)

plot(density(rho[,1]), col = 2, lwd = 2, ylim = c(0, 4), main = "rho estimation")
points(density(rho[,2]), col = 3, type = "l", lwd = 2)
points(density(rho[,3]), col = 4, type = "l", lwd = 2)
legend("topleft", c("MLE", "LSE", "Y-W"), col = 2:4, lwd = 2)

plot(density(sigma[,1]), col = 2, lwd = 2, ylim = c(0, 4), main = "sigma estimation")
points(density(sigma[,2]), col = 3, type = "l", lwd = 2)
points(density(sigma[,3]), col = 4, type = "l", lwd = 2)
legend("topleft", c("MLE", "LSE", "Y-W"), col = 2:4, lwd = 2)
```

![Estimated parameters - 100 times simulation](/assets/Statistical Signal Processing/2.png)

Final average output is below

|**Model**| |$\hat{\mu}$(100 times) | |$\hat{\rho}$(100 times) | |$\hat{\sigma}$(100 times) |
|:---:|-|:----------------:|-|:----------------:|-|:----------------:|
|**True**| |2 || 0.2 | |1.414214 |
|**MLE**| |1.988347 | |0.1753806 | |1.414081 |
|**LSE**| |1.988119 | |0.1746358 | |1.434091 |
|**Y-W**| |1.985870 | |0.1748110 | |1.470657 |

The contents in parentheses are variance. 
We can easily check 3-methods are estimate true parameters well.

## 4. Other topic

### 4-1. Power of sample mean

Two of questions arise

- Not using iterative method in MLE or LSE

- How powerful moment estimator sample mean $\bar{X}$

then think about LSE method. When select moment estimator $\hat{\mu} = \bar{X}$,

$$
S(\rho) = \sum_{t = 1}^n(X_t - \bar{X} - \rho(X_{t-1} - \bar{X}))^2
$$

then

$$
\hat{\rho} = \frac{\sum_{t = 1}^n (X_{t - 1} - \bar{X})(X_{t} - \bar{X})}{\sum_{t = 1}^n(X_{t - 1} - \bar{X})^2}
$$

It is same with Yule-Walker estimation. Showing these estimated parameters $(\hat{\mu}, \hat{\rho})^t$ is powerful.
Using the facts

$$
\begin{split}
Var(X_0) &= \frac{\sigma^2}{1 - \rho^2}\\
Var(X_1) &= \frac{\rho^2\sigma^2}{1 - \rho^2} + \sigma^2 = \frac{\sigma^2}{1 - \rho^2}
\end{split}
$$

$$
\begin{split}
Var(X_i) &= E\left[ (X_i - \mu)^2 \right] \\
&= E\left[ (\rho (X_{i-1} - \mu) + e_i)^2 \right] \\
&= E\left[ \rho^2 (X_{i - 1} - \mu)^2 - 2\rho(X_{i - 1} - \mu)e_i + e_i^2 \right] \\
&= \rho^2 Var(X_{i - 1}) + Var(e_i) \\
&= \rho^2 Var(X_{i - 1}) + \sigma^2 \\
&= \frac{\sigma^2}{1 - \rho^2}\quad (\because Var(X_0) = \sigma^2/(1 - \rho^2), Var(X_1) = \rho^2\sigma^2/(1 - \rho^2) + \sigma^2 = \sigma^2/(1 - \rho^2))
\end{split}
$$

$$
\begin{split}
Cov(X_i, X_{i + h}) &= Cov(X_i, \mu + \rho(X_{i + h}) - \mu + e_{i + h -1}) \\ &= \rho Cov(X_i, X_{i + h - 1}) = \rho Cov(X_i, \mu + \rho(X_{i + h - 2} - \mu) + e_{i + h -2})\\
&= \rho^2 Cov(X_i, X_{i + h - 2}) = \rho^3 Cov(X_i, X_{i + h -3}) = \cdots\\
&= \rho^h Cov(X_i, X_{i + h - h}) \\
&= \rho^h Var(X_i) \\
&= \frac{\rho^h}{1 - \rho^2}\sigma^2
\end{split}
$$

$$
\begin{split}
Var(\bar{X}) &= \frac{1}{n^2}Var(\sum_{t = 1}^nX_t) \\
&= \frac{1}{n^2}\left[\sum_{t = 1}^n Var(X_t) + 2\sum_{t = 1}^{n - 1}\sum_{h = 1}^{n - t}Cov(X_t, X_{t + h})\right] \\
&= \frac{1}{n^2}\frac{\sigma^2}{1 - \rho^2}\left[ n + 2\sum_{t = 1}^{n - 1}\sum_{h = 1}^{n - t}\rho^h \right] \\
&= \frac{1}{n^2}\frac{\sigma^2}{1 - \rho^2}\left[ n + 2\sum_{t = 1}^{n - 1}\frac{\rho(1 - \rho^{n - t})}{1 - \rho} \right]\\
&= \frac{1}{n^2}\frac{\sigma^2}{1 - \rho^2}\left[ n + \frac{2\rho}{1 - \rho}\left(n - 1-\sum_{t = 1}^{n - 1}\rho^{n - t} \right)\right]\\
&= \frac{1}{n^2}\frac{\sigma^2}{1 - \rho^2}\left[ n + \frac{2\rho}{1 - \rho}\left(n - 1 -\frac{\rho(1 - \rho^{n - 1})}{1 - \rho} \right)\right]\\
\end{split}
$$

Also asymptotic variance of $\bar{X}$ is calculated by $Chebyshev-Inequality$.

$$
\begin{split}
P(|\bar{X} - \mu| > \epsilon) = P((\bar{X} - \mu)^2 > \epsilon^2) \le \frac{Var(\bar{X})}{\epsilon} \rightarrow 0
\end{split}
$$

Thus if $n \rightarrow \infty$, 

$$
\sqrt{n}(\bar{X} - \mu) \sim N(0, \frac{\sigma^2}{1 - \rho^2})
$$

Also by M-estimation,

$$
\sqrt{n}(\hat{\rho} - \rho) \sim N(0, [E(X_0^2)]^{-1}\sigma^2)
$$

### 4-2. Check loss

Quantile regression is a type of regression analysis used in statistics and econometrics. 
Whereas the method of least squares results in estimates of the conditional mean of the response variable given certain values of the predictor variables, quantile regression aims at estimating either the conditional median or other quantiles of the response variable. 
Also LSE use squared loss but Quantile regression use check loss like below.

![Figure4. Check loss function](/assets/Statistical Signal Processing/3.png)

using this, we can adjust LSE via check loss.

$$
\begin{split}
Q(\rho, \mu, \tau) &= \sum_{t = 1}^p \rho_{\tau}(X_t - \mu - \rho(X_{t - 1} - \mu))\\
&= \sum_{t = 1}^p \rho_{\tau}(X_t - q)\\
&= \left[(\tau - 1)\sum_{X_t < q}(X_t - q)+ \tau\sum_{X_t \ge q}(X_t - q)\right]
\end{split}
$$

When we use quantile $\tau = 0.5$, as called median regression

```r
rm(list = ls())

library(tensorflow)
set.seed(1)
n <- 50
mu <- 2
sigma <- sqrt(2)
rho <- 0.2
x0 <- rnorm(1, mu, sigma/sqrt(1 - rho^2))
xi <- c(x0, rep(NA, n))
for(i in 1:n){
  past <- xi[i]
  past <- mu + rho*(past - mu) + rnorm(1, 0, sigma)
  current <- past
  xi[i + 1] <- current
}
xt <- as.matrix(xi[2:51])
xt1<- as.matrix(xi[1:50])

fit.function <- function(x0, xt, xt1){
  
  sess = tf$Session()
  X_0 = tf$placeholder(shape = shape(), dtype = tf$float32)
  X_t = tf$placeholder(shape = shape(n, 1), dtype = tf$float32)
  X_t1 = tf$placeholder(shape = shape(n, 1), dtype = tf$float32)
  rho = tf$Variable(0.1)
  mu = tf$Variable(2.0)
  c = mu*(1 - rho)
  tau = tf$constant(0.5)
  
  cost1 = tau*tf$maximum((X_t - c - rho*X_t1), 0)
  cost2 = (1 - tau)*tf$maximum(-(X_t - c - rho*X_t1), 0)
  cost = tf$reduce_sum(cost1 + cost2)
  train_step = tf$train$AdamOptimizer()$minimize(cost)
  init = tf$global_variables_initializer()
  sess$run(init)
  criteria <- 1e-8
  past <- Inf
  
  for(i in 1:20000){
    sess$run(train_step,feed_dict=dict(X_0=x0, X_t = xt, X_t1= xt1))
    current <- sess$run(cost,feed_dict=dict(X_0= x0, X_t = xt, X_t1= xt1))
    if(abs(current - past)< criteria) break
    past <- current
  }
  cost1.val = pmax(0.5*(xt - sess$run(c) - sess$run(rho)*xt1), 0)
  cost2.val = pmax(-0.5*(xt - sess$run(c) - sess$run(rho)*xt1), 0)
  sigma = sqrt(1/(length(xt))*sum(cost1.val + cost2.val))
  return(list(mu = sess$run(mu), rho = sess$run(rho), sigma = sigma))
}

fit <- fit.function(x0, xt, xt1)

num.simul <- 100
mu.matrix <- matrix(NA, num.simul, 1)
rho.matrix <- matrix(NA, num.simul, 1)
sigma.matrix <- matrix(NA, num.simul, 1)

for(iter in 1:num.simul){
  n <- 50
  mu <- 2
  sigma <- sqrt(2)
  rho <- 0.2
  x0 <- rnorm(1, mu, sigma/sqrt(1 - rho^2))
  xi <- c(x0, rep(NA, n))
  for(i in 1:n){
    past <- xi[i]
    past <- mu + rho*(past - mu) + rnorm(1, 0, sigma)
    current <- past
    xi[i + 1] <- current
  }
  xt <- as.matrix(xi[2:51])
  xt1<- as.matrix(xi[1:50])
  
  fit <- fit.function(x0, xt, xt1)
  
  mu.matrix[iter,1] <- fit$mu
  rho.matrix[iter,1] <- fit$rho
  sigma.matrix[iter,1] <- fit$sigma
  
}

par(mfrow = c(1, 3))

plot(density(mu.matrix), col = 2, lwd = 2, ylim = c(0, 2), main = "mu quantile estimation")
plot(density(rho.matrix), col = 2, lwd = 2, ylim = c(0, 4), main = "rho quantile estimation")
plot(density(sigma.matrix), col = 2, lwd = 2, ylim = c(0, 11), main = "sigma quantile estimation")
```

![Figure5. Check loss estimation](/assets/Statistical Signal Processing/4.png)

|**Model**||$\hat{\mu}$||$\hat{\rho}$||$\hat{\sigma}$|
|:-------:|-|:---------:|-|:----------:|-|:------------:|
|**True**||2||0.2||1.414214|
|**Check loss**||1.961759||0.1801444|| 0.7502005|
|**(Variance)**||(0.0703)||(0.0235)||(0.00144)|

When using check loss, rho made better estimates than the above three methods. 
However, the estimates of mu and sigma came with very bad results. 
The good news is that the variance of the estimated sigma is less than the above three methods, and this method is robust.

## 5. Conclusion

We looked at three estimation methods and saw performance of sample mean $\bar{X}$. 
In this paper, we have only looked at AR(1) model. 
The AR(p) model is more complex and even the ARMA(p, q) model is more complex. 
There is also the point that the AR(p, q) model can not use Yule-Walker method. 
However, it was very interesting and difficult estimate using the method learned in the class in the AR(1) model. 
Also I think it is good to be able to confirm that the use of check loss is applicable to many other models in the future.

## References

H. Ltkepohl, New Introduction to Multiple Time Series Analysis, Springer Publishing Company, Incorporated, 2007.

J. A. a. Y. Tang, tensorflow: R Interface to 'TensorFlow', https://CRAN.R-project.org/package=tensorflow, 2019.

R. Koenker, Galton, Edgeworth, Frisch, and prospects for quantile regression in economics, UIUC.edu., 1998.

S. M. Kay, Fundamentals of Statistical Signal Processing: Estimation Theory, Prentice-Hall, Inc., 1993.
