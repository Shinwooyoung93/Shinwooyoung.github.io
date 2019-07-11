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
# array([[0.66666667, 1.        ],
#        [0.33333333, 0.55555556],
#        [1.        , 0.44444444]])
```

We let input variable size as 2 in input layer and hidden node size as 3 in hidden layer.

```python
inputSize = 2
outputSize = 1
hiddenSize1 = 3
hiddenSize2 = 3
```

Setting weights and initial weights are zeros.

```python
W1 = np.zeros([inputSize, hiddenSize1]) + 0.1
W2 = np.zeros([hiddenSize1, hiddenSize2]) + 0.1
W3 = np.zeros([hiddenSize2, outputSize]) + 0.1

print("W1: \n" + str(W1))
print("W2: \n" + str(W2))
print("W3: \n" + str(W3))
```
```python
# W1: 
# [[0.1 0.1 0.1]
#  [0.1 0.1 0.1]]
# W2: 
# [[0.1 0.1 0.1]
#  [0.1 0.1 0.1]
#  [0.1 0.1 0.1]]
# W3: 
# [[0.1]
#  [0.1]
#  [0.1]]
```

Define activation function and derivatived activation function.

```python
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidPrime(x):
    return x * (1 - x)
```

Define "feedforward pass" function using $numpy.matmul$.

```python
def forward(X):
    a1 = np.matmul(X, W1) 
    a2 = np.matmul(sigmoid(a1), W2)
    a3 = np.matmul(sigmoid(a2), W3)
    o = sigmoid(a3) 
    return o 

print("predicted output: " + str(forward(X).T))
print("loss: " + str(np.mean(np.square(t - forward(X)))))
```
```python
# predicted output: [[0.54045105 0.54034353 0.54042039]]
# loss: 0.12617687296377364
```

This output means probability used by sigmoid function. Also see back-propagation process that used to update error.

```python
# "feedforward pass"
a1 = np.matmul(X, W1) 
a2 = np.matmul(sigmoid(a1), W2)
a3 = np.matmul(sigmoid(a2), W3)
o = sigmoid(a3) 

# doing back-propagation
o_delta = -(t - o)*sigmoidPrime(o)
a2_delta = np.matmul(o_delta, W3.T)*sigmoidPrime(a2)
a1_delta = np.matmul(a2_delta, W2.T)*sigmoidPrime(a1) 

# doing update by back-propagation
W1 -= np.matmul(X.T, a1_delta)
W2 -= np.matmul(a1.T, a2_delta)
W3 -= np.matmul(a2.T, o_delta)

print("W1: \n" + str(W1))
print("W2: \n" + str(W2))
print("W3: \n" + str(W3))
```

```python
# W1: 
# [[0.10008983 0.10008983 0.10008983]
#  [0.10009367 0.10009367 0.10009367]]
# W2: 
# [[0.1004883 0.1004883 0.1004883]
#  [0.1004883 0.1004883 0.1004883]
#  [0.1004883 0.1004883 0.1004883]]
# W3: 
# [[0.14122566]
#  [0.14122566]
#  [0.14122566]]
```

Using these weights again to "feedforward pass". 
This works are possible to send error back and reapply error and when increase epochs, reduced error gradually.

```python
class Neural_Network(object):
    
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize1 = 3
        self.hiddenSize2 = 3
        
        self.W1 = np.zeros([self.inputSize, self.hiddenSize1]) + 0.1
        self.W2 = np.zeros([self.hiddenSize1, self.hiddenSize2]) + 0.1
        self.W3 = np.zeros([self.hiddenSize2, self.outputSize]) + 0.1
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoidPrime(self, x):
        return x * (1 - x)
  
    def forward(self, X):
        self.a1 = np.matmul(X, self.W1) 
        self.a2 = np.matmul(self.sigmoid(self.a1), self.W2)
        self.a3 = np.matmul(self.sigmoid(self.a2), self.W3)
        o = self.sigmoid(self.a3) 
        return o 
   
    def backward(self, X, t, o):
        self.o_delta = -(t - o)*self.sigmoidPrime(o)
        self.a2_delta = np.matmul(self.o_delta, 
                                  self.W3.T)*self.sigmoidPrime(self.a3) 
        self.a1_delta = np.matmul(self.a2_delta, 
                                  self.W2.T)*self.sigmoidPrime(self.a2) 

        self.W1 -= np.matmul(X.T, self.a1_delta) 
        self.W2 -= np.matmul(self.a1.T, self.a2_delta)
        self.W3 -= np.matmul(self.a2.T, self.o_delta)
  
    def train(self, X, t):
        o = self.forward(X)
        self.backward(X, t, o)

NN = Neural_Network()

for i in range(10000):
    if i % 1000 ==0 or i == 1:
        print("predicted output: " + str(NN.forward(X).T) )
        print("loss: " + str(np.mean(np.square(t - NN.forward(X)))))
        print("\n")
    NN.train(X, t)
```
```python
#predicted output: [[0.54045105 0.54034353 0.54042039]]
#loss: 0.12617687296377364


#predicted output: [[0.5570245  0.55687276 0.55698123]]
#loss: 0.1150369188377854


#predicted output: [[0.72266845 0.72262319 0.72265611]]
#loss: 0.0337720657418149


#predicted output: [[0.73006422 0.73006664 0.73006488]]
#loss: 0.03142134754185451


#predicted output: [[0.73105931 0.73105979 0.73105944]]
#loss: 0.031114001170512426


#predicted output: [[0.73106591 0.73106592 0.73106591]]
#loss: 0.031112000677230422


#predicted output: [[0.73105902 0.73105902 0.73105902]]
#loss: 0.03111412332216862


#predicted output: [[0.73105855 0.73105855 0.73105855]]
#loss: 0.03111426688467415


#predicted output: [[0.73105857 0.73105857 0.73105857]]
#loss: 0.031114259987152403


#predicted output: [[0.73105858 0.73105858 0.73105858]]
#loss: 0.031114258512887197


#predicted output: [[0.73105858 0.73105858 0.73105858]]
#loss: 0.03111425847414064
```

We have previously seen that the back-propagation, and the reason why we continually update the error by propagating it backwards is because the error affects the weight adjustment to give better results through the neural network. However, the more complicated the structure, the more efficient it is. The gradient descent method is a method designed to do this efficiently because it takes a long time to calculate all the weight combinations in the neural network.
