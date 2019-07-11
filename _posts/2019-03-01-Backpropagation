---
layout: post
title:  "Backpropagation"
date:   2019-03-01
use_math: true
tags:
 - Python
 - english
 - Research
---

# Introduction

When we compute the derivative of loss function with a large amount of layers, it can be implemented efficiently on a parallel architecture. 
Back-propagation method enable this and made fitting a complex model computationally manageable.

# Setting structure

## Notations

We assume notations that

* activation funtion : $h(\cdot)$ <br/>

* input variable(input layer) : $x_i \quad i = 1, \ldots, K_0$ <br/>

* 1-th hidden node(hidden layer) : $h(a_j^{(1)}) \quad j = 1, \ldots, K_1$ <br/>

* 2-th hidden node(hidden layer) : $h(a_k^{(2)}) \quad k = 1, \ldots, K_2$ <br/>

* predicted output(output layer) : $P = \sigma (a^{(3)})$<br/>

* ture variable : $t_n \quad n = 1, \ldots, N$

then, see each layer's node used weight.

$$
\begin{split}
a_j^{(1)} &= \sum_{i = 1}^{K_0} w_{ji}^{(1)} x_i + w_{j0}^{(1)} \quad j = 1, \ldots, K_1 \\
a_k^{(2)} &= \sum_{j = 1}^{K_1} w_{kj}^{(2)} h(a_j^{(1)}) + w_{k0}^{(2)} \quad k = 1, \ldots, K_2 \\
a^{(3)} &= \sum_{k = 1}^{K_2} w_{k}^{(3)} h(a_k^{(2)}) + w_{0}^{(3)}
\end{split}
$$

Our goal is minimize error function that 

$$
E = \sum_{n = 1}^N E_n = \frac{1}{2}\sum_{n = 1}^N ||t_n - P_n ||^2 \Longrightarrow \nabla E = \sum_{n = 1}^N \nabla E_n = - \sum_{n = 1}^N(t_n - P_n)
$$

By calculating backwards, gradients of all weight can be calculated easily.

$$
\begin{split}
\delta^{(3)} &= \frac{\partial E_n}{\partial a^{(3)}} = \frac{\partial E_n}{\partial P_n} \cdot \frac{\partial P_n}{\partial a^{(3)}} = -(t_n - P_n) \cdot \sigma^{'}(a^{(3)}) \\
\frac{\partial E_n}{\partial w_k^{(3)}} &= \frac{\partial E_n}{\partial a^{(3)}} \cdot \frac{\partial a^{(3)}}{\partial w_k^{(3)}} = \delta^{(3)} \cdot h(a^{(2)}) \\
\delta_k^{(2)} &= \frac{\partial E_n}{\partial a_k^{(2)}} = \frac{\partial E_n}{\partial a^{(3)}}\cdot \frac{\partial a^{(3)}}{\partial a_k^{(2)}} = \delta^{(3)} \cdot w_k^{(3)} \cdot h^{'}(a_k^{(2)}) \\
\frac{\partial E_n}{\partial w_{kj}^{(2)}} &= \frac{\partial E_n}{\partial a_k^{(2)}} \cdot \frac{\partial a_k^{(2)}}{\partial w_{kj}^{(2)}} = \delta_k^{(2)} \cdot h(a_j^{(1)}) \\
\delta_j^{(1)} &= \frac{\partial E_n}{\partial a_j^{(1)}} = \sum_{k = 1}^{K_2}\frac{\partial E_n}{\partial a_k^{(2)}}\cdot \frac{\partial a_k^{(2)}}{\partial a_j^{(1)}} = \sum_{k = 1}^{K_2} \delta_k^{(2)} \cdot w_{kj}^{(2)} \cdot h^{'}(a_j^{(1)}) \\
\frac{\partial E_n}{\partial w_{ji}^{(1)}} &= \delta_j^{(1)} \cdot x_i
\end{split}
$$

Important thing is these gradients made by "feedforward pass" and initial gradients.

## Python code

```python
%reset -f
import numpy as np

X = np.array(([2, 9], [1, 5], [3, 4]))
X = X / np.amax(X, axis=0)
t = np.array(([2.9], [2.3], [2.5]))/2.9
X
```
```python
array([[0.66666667, 1.        ],
       [0.33333333, 0.55555556],
       [1.        , 0.44444444]])
```


