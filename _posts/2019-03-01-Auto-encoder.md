---
layout: post
title:  "Auto-encoder"
date:   2019-03-01
use_math: true
tags:
 - Python
 - english
 - Research
---

# Introduction

An auto-encoder is a type of nonlinear principal-component decomposition(PCA). 
It can also be said to play a role in capturing the characteristics of massive data. 

# Setting structure

## Notations

Encoder means that input variables convert to internal expressions, and decoder means internal expressions convert to output variables. 
Auto-encoder process is symmetric based on hidden layer. Thus, we want to make below equation.

* input variable(mean is 0) : $x_i \in \mathbb{R}^p$
* uncorrelated features : $z_i \in \mathbb{R}^q, \: q \le p$
* orthonormal weight matrix : $W \in \mathbb{R}^{q \times p}$  that $z_i = W x_i$

$$
\begin{split}
\text{minimize}_{W \in \mathbb{R}^{q \times p}} \sum_{i = 1}^n ||x_i - W^{'}g(Wx_i)||^2_2
\end{split}
$$

## Explore image

```python
%reset -f
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(train_X, train_t), (test_X, test_t) = tf.keras.datasets.mnist.load_data()
valid_X, train_X = np.array(train_X[:5000]), np.array(train_X[5000:])

image = train_X[0]
fig = plt.figure(figsize = (5, 5))
X_image = np.array(image, dtype = 'float')
plt.imshow(X_image.reshape((28, 28)))
```

![](/assets/Auto-encoder/1.png)

We check how above image is changed by auto-encoder.

## Normalizing input data

```python
train_X = train_X.reshape(-1, 28*28)/255.0
valid_X = valid_X.reshape(-1, 28*28)/255.0
```

## Define batch function

class Batch:

    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    def next_batch(self,batch_size,shuffle = True):
        
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self._data = self.data[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples) 
            np.random.shuffle(idx0)  
            self._data = self.data[idx0] 

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples 
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]              
            return np.concatenate((data_rest_part,
                                   data_new_part),
                                  axis=0), self._epochs_completed
        
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch            
            return self._data[start:end], self._epochs_completed
            
## Select optimize algorithm

When we try image data processing, simple optimize algorithm can produce wrong output.Typically, Stochatic gradient descent method(SGD) is widely used because it is simple. 
SGD's convergence theory is widely known by L.Bottou(1998).

$$
\begin{split}
& 1. L(\theta) \text{ can be differentiated 3 times, and that is also continous function.} \\
& 2. E\big[ ||H(\theta, D||^k \big] = O\big(\theta^k \big), \quad k = 2, 3, 4 \\
& 3. \text{exist constant }C > 0\text{ that satisfy } \inf_{||\theta|| \ge C} \theta^t \nabla L(\theta) > 0
\end{split}
$$

If satisfy above conditions then probability converge to local minimum using SGD method with learning rate $\epsilon_t \propto \frac{1}{t}$.

But because of vanishing gradients problem and converge to local minimum problem, Adam(Adaptive Moment estimation) optimize method is used widely. That is combined by RMSprop and Momentum method.
Similar to the Momentum method, it stores the exponential average of the slope calculated so far, and stores the exponential average of the squared value of the slope, similar to RMSprop. We just look the Adam algorithm and choose the algorithm. 

**Algorithm** : The Adam algorithm

**Require** : Step size $\epsilon$ (Suggested default : 0.001)

**Require** : Exponential decay rates for moment estimates, $\rho_1$ and $\rho_2$ in $[0, 1)$ (Suggested default : 0.9 and 0.999 respectively)

**Require** : Small constant $\delta$ used for numerical stabilization. (Suggested default : $10^{-8}$)

**Require** : Initial parameters $\theta$

1. Initialize 1st and 2nd moment variables $s = 0, r = 0$
2. Initialize time step $t = 0$
3. **while** stopping criterion not met **do**
4. $\quad$Sample a minibatch of $m$ examples from the training set $\{x^{(1)}, \ldots, x^{(m)}\}$ with targets $y^{(i)}$.
5. $\quad$Compute gradient: $g \leftarrow \frac{1}{m}\nabla_{\theta}\sum_i L(f(x^{(i)};\theta), y^{i})$.
6. $\quad$$t \leftarrow t + 1$.
7. $\quad$Update biased first moment estimate : $s \leftarrow \rho_1\cdot s + (1 - \rho_1)\cdot g$.
8. $\quad$Update biased second moment estimate : $r \leftarrow \rho_2\cdot r + (1 - \rho_2)\cdot g\odot g$.
9. $\quad$Correct bias in first moment : $\hat{s} \leftarrow \frac{s}{1 - \rho_1^t}$.
10. $\quad$Correct bias in second moment : $\hat{s} \leftarrow \frac{r}{1 - \rho_2^t}$.
11. $\quad$Correct update : $\nabla \theta = -\epsilon \frac{\hat{s}}{\hat{r} + \delta}$ (operations applied element-wise) 
12. $\quad$Apply update : $\theta \leftarrow \theta + \nabla \cdot \theta$
13. **end while**

## Define auto-encoder function

```python
num_features = 28 * 28
num_units1 = 300  # encoder
num_units2 = 150  # coding units
num_units3 = num_units1  # decoder
num_output = num_features # reconstruction

lamb = 0.0001
batch_size = 1000

X = tf.placeholder(tf.float32, shape = [None, num_features])
W1 = tf.Variable(tf.truncated_normal([num_features, num_units1]))
b1 = tf.Variable(tf.zeros([num_units1]))
hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

W3 = tf.transpose(W2)
b3 = tf.Variable(tf.zeros([num_units3]))
hidden3 = tf.nn.tanh(tf.matmul(hidden2, W3) + b3)

W4 = tf.transpose(W1)
b4 = tf.Variable(tf.zeros([num_output]))
t = tf.nn.tanh(tf.matmul(hidden3, W4) + b4)

cost = tf.reduce_mean(tf.square(X - t))
regularizers = tf.nn.l2_loss(W1) +  tf.nn.l2_loss(W2)
loss = cost + lamb*regularizers

train_step = tf.train.AdamOptimizer().minimize(loss)

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mse_val = []

i = 0
for _ in range(5000):
    i += 1    
    X_batch, epoch = Batch(train_X).next_batch(batch_size)
    _, loss_val = sess.run([train_step, loss], feed_dict = {X: X_batch})
    if i % 500 ==0 or i == 1:
        mse = sess.run(cost, feed_dict = {X: X_batch})
        mse_val.append(mse)
        print('step:%d,loss:%f,mse:%f' %(i,loss_val,mse))
```
```python
# step:1,loss:11.762371,mse:0.938995
# step:500,loss:4.368691,mse:0.000001
# step:1000,loss:1.665577,mse:0.000001
# step:1500,loss:0.576825,mse:0.000001
```

# Validation

```python
pred_image = sess.run(t, feed_dict = {X: train_X[0:1]})
true_image = np.array(train_X[0], dtype = 'float').reshape((28, 28))
pred_image = np.array(pred_image, dtype = 'float').reshape((28, 28))

fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.imshow(true_image)
ax1.set_title("True image", fontsize = 15)
ax2.imshow(pred_image)
ax2.set_title("pred image", fontsize = 15)
plt.savefig('2.png', dpi=300)
```

![](/assets/Auto-encoder/2.png)
