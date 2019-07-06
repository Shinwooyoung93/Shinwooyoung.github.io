---
layout: post
title:  "Research format"
date:   2019-07-06
use_math: true
tags:
 - python
---

# [Title]

*KU, Shin wooyoung(your name)*

## 1. Introduction

The introduction provide a theoretical, practical, and/or historical background so that your reader will be able to understand what you did and why it was worth doing with 2-4 lines.

## 2. Materials and Methods

Here you describe the proposed model and all the theoretical representations and derivations of the model. You can also describe what the model implicates by providing relevant interpretations.

* $\tilde{\phi}(x_k) = \phi(x_k) - \frac{1}{n}\sum_{k = 1}^n\phi(x_k)$


* $\Sigma = \frac{1}{n}\sum_{i = 1}^n \phi(x_i)\phi(x_i)^t \Rightarrow \Sigma v = \lambda v$

$$
\begin{split}
\tilde{K(x_i, x_j)} &= \tilde{\phi}(x_i)^t\tilde{\phi}(x_j)\\
&= \left(\phi(x_i) - \frac{1}{n}\sum_{k = 1}^n\phi(x_k)\right)^t\left(\phi(x_j) - \frac{1}{n}\sum_{k = 1}^n\phi(x_k)\right)\\
&= \phi(x_i)^t\phi(x_j) - \frac{1}{n}\sum_{k = 1}^n\phi(x_i)^t\phi(x_k) - \frac{1}{n}\sum_{k = 1}^n\phi(x_k)^t\phi(x_j) + \frac{1}{n^2}\sum_{k = 1}^n\sum_{l = 1}^n\phi(x_l)^t\phi(x_k)\\
&= K(x_i, x_j) - \frac{1}{n}\sum_{k = 1}^nK(x_i, x_k) - \frac{1}{n}\sum_{k = 1}^nK(x_j, x_k) + \frac{1}{n^2}\sum_{k = 1}^n\sum_{l = 1}^nK(x_l, x_k)\\
&\therefore \tilde{K} =  K - 2\textbf{1}_{1/n}K + \textbf{1}_{1/n}K\textbf{1}_{1/n}
\end{split}
$$

### 2-a. Subtitle

### 2-b. Subtitle

## 3. Code

```python
%reset -f
```

### 3-a. Data preparation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops

(X_train, t_train), (X_test, t_test) = tf.keras.datasets.mnist.load_data()
X_train= np.reshape(X_train, [-1, 784])
X_test= np.reshape(X_test, [-1, 784])
train_t= np.array(pd.get_dummies(t_train))*1.
test_t = np.array(pd.get_dummies(t_test))*1.

fig = plt.figure(figsize = (8,8))
for i in range(10):
    c = 0
    for (image, label) in zip(np.reshape(X_train, [-1, 28,28]), train_t):
        if np.argmax(label) != i: continue
        subplot = fig.add_subplot(10,10,i*10+c+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,
                       cmap=plt.cm.gray_r, interpolation="nearest")
        c += 1
        if c == 10: break
            
#plt.savefig('Report_format1.png', dpi=300)
```

![]("/_images/Report_format/Report_format1.png")

```python
from sklearn.datasets import make_circles

X, y = make_circles(n_samples = 200, random_state = 1, noise = 0.1, factor = 0.2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.title('True plot', size = 15)
plt.savefig('1.png', dpi = 300)
plt.show()
#plt.savefig('Report_format2.png', dpi=300)
```

![]("/_images/Report_format/Report_format2.png")

### 3-b. Simulation

## 4. Discussion

Interpret the simulation results and draw conclusions by comparing with (1) what others have seen, or (2) what might have been expected in light of theory or hypothesis.

