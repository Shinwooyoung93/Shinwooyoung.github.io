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

Thus AR(1) model's likelihood function is $f(X_0, \ldots, X_i) = f(X_0)f(X_1|X_0)\ldots f(X_n|X_{n - 1})$ and $X_t = c + \rho X_{t - 1} + e_t$.

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



