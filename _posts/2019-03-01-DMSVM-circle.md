---
layout: post
title:  "DMSVM - cricle"
date:   2019-03-01
use_math: true
tags:
 - Python
 - english
 - Research
---

# Introduction

We see that true circle data.

# Simulation

## Datasets

```python
%reset -f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from pandas import DataFrame
import sklearn
from sklearn.datasets import make_circles

X, t = sklearn.datasets.make_circles(n_samples = 1000, shuffle = True,
                                     noise = 0.1, random_state = 1, factor = 0.6)
df = DataFrame(dict(x1 = X[:,0], x2 = X[:,1], t = t))
df1= df.loc[df['t'] == 1]
df2= df.loc[df['t'] == 0]

fig = plt.figure(figsize = (6, 6))
subplot = fig.add_subplot(1, 1, 1)
subplot.set_ylim([min(df["x2"]) -0.1, max(df["x2"]) + 0.1])
subplot.set_xlim([min(df["x1"]) -0.1, max(df["x1"]) + 0.1])
subplot.scatter(df1["x1"], df1["x2"], marker = "x")
subplot.scatter(df2["x1"], df2["x2"], marker = "o")
subplot.set_title("True plot", fontsize = 20)
plt.savefig('1.png', dpi=300)
```

![](/assets/DMSVM-circle/1.png)

## Linear SVM

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split 
X_train,X_test,t_train,t_test= np.array(train_test_split(df[["x1","x2"]],
                                                         df.iloc[:,2:3],random_state=0))
                                                         
test_t1 = np.array(pd.get_dummies(t_test.astype("category")))
train_t1= np.array(pd.get_dummies(t_train.astype("category")))

test_bind = pd.concat([X_train, t_train], axis = 1)
X_test1 = test_bind.loc[test_bind["t"]==0,]
X_test1 = X_test1.iloc[:,0:2]
X_test2 = test_bind.loc[test_bind["t"]==1,]
X_test2 = X_test2.iloc[:,0:2]

test_t2 = np.where(test_t1 == 0, 1, 0)
train_t2= np.where(train_t1== 0, 1, 0)

num_factors = 2
num_features = 2

X = tf.placeholder(tf.float32, shape=[None, num_features])
w1 = tf.Variable(tf.truncated_normal([num_features, num_factors]))
b1 = tf.Variable(tf.zeros([num_factors]))
C = tf.constant([0.1])

prob = tf.matmul(X, w1) + b1
t1 = tf.placeholder(tf.float32, [None, num_factors])
t2 = tf.placeholder(tf.float32, [None, num_factors])
cost = tf.reduce_sum(tf.square(w1))
hinge1 = tf.reduce_sum(tf.multiply(prob, t1), axis = 1)
hinge2 = tf.multiply(prob, t2)
hinge2 = tf.boolean_mask(hinge2, tf.not_equal(t2, 0.), axis = 0)
hinge2 = tf.reshape(hinge2, [-1, num_factors-1])
hinge2 = tf.reduce_max(hinge2, axis = 1)
hinge = tf.maximum(1 - hinge1 + hinge2, 0)
loss = tf.add(cost, tf.multiply(C, tf.reduce_sum(tf.square(hinge))))
train_step = tf.train.AdamOptimizer().minimize(loss)

output = tf.argmax(prob, axis = 1)
correct_prediction = tf.equal(output, tf.argmax(t1, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_accuracy1= []
test_accuracy1 = []

i = 0

for _ in range(10001):
    i += 1
    _, loss_val = sess.run([train_step, loss], 
                           feed_dict={X:X_train,t1:train_t1,t2:train_t2})
    if i % 1000 == 0 or i == 1:
        train_acc = sess.run(accuracy, 
                             feed_dict={X:X_train,t1:train_t1,t2:train_t2})
        train_accuracy1.append(train_acc)
        test_acc = sess.run(accuracy, 
                            feed_dict={X:X_test,t1:test_t1,t2:test_t2})
        test_accuracy1.append(test_acc)
        print('step:%d,loss:%f,Train:%f,Test:%f' 
              %(i,loss_val,train_acc,test_acc))

# step:1,loss:134.007095,Train:0.498667,Test:0.512000
# step:1000,loss:75.653046,Train:0.493333,Test:0.468000
# step:2000,loss:75.357826,Train:0.476000,Test:0.460000
# step:3000,loss:75.233871,Train:0.478667,Test:0.456000
# step:4000,loss:75.105072,Train:0.480000,Test:0.456000
# step:5000,loss:75.009033,Train:0.480000,Test:0.456000
# step:6000,loss:74.965454,Train:0.477333,Test:0.456000
# step:7000,loss:74.956299,Train:0.477333,Test:0.452000
# step:8000,loss:74.955719,Train:0.477333,Test:0.452000
# step:9000,loss:74.955719,Train:0.477333,Test:0.452000
# step:10000,loss:74.955711,Train:0.478667,Test:0.452000

grid1 = []
for x2 in np.linspace(min(df["x2"])-0.1,max(df["x2"])+0.1, 500):
    for x1 in np.linspace(min(df["x1"])-0.1,max(df["x1"])+0.1, 500):
        grid1.append((x1, x2))
out_val1 = sess.run(output, feed_dict = {X: grid1})
out_val1 = pd.get_dummies(out_val1)
out_val11= np.array(out_val1.iloc[:, 0:1]).reshape((500, 500))
out_val12= np.array(out_val1.iloc[:, 1:2]).reshape((500, 500))

fig = plt.figure(figsize = (6, 6))
subplot = fig.add_subplot(1, 1, 1)
min_val1 = min(X_test["x1"]) - 0.1
max_val1 = max(X_test["x1"]) + 0.1
min_val2 = min(X_test["x2"]) - 0.1
max_val2 = max(X_test["x2"]) + 0.1
subplot.set_ylim([min_val2, max_val2])
subplot.set_xlim([min_val1, max_val1])
subplot.scatter(X_test1["x1"], X_test1["x2"], marker = "x")
subplot.scatter(X_test2["x1"], X_test2["x2"], marker = "o")
subplot.set_title("Predict SVM", fontsize = 20)
subplot.imshow(out_val11, origin = "lower", 
               extent = (min_val1, max_val1, min_val2, max_val2),
               cmap = plt.cm.gray_r, alpha = 0.3)
subplot.imshow(out_val12, origin = "lower", 
               extent = (min_val1, max_val1, min_val2, max_val2),
               cmap = plt.cm.gray_r, alpha = 0.1)
plt.savefig('2.png', dpi=300)
```

![](/assets/DMSVM-circle/2.png)

Linear SVM only split with linear line.

## Neural network

```python
test_t = np.array(pd.get_dummies(t_test.astype("category")))
train_t= np.array(pd.get_dummies(t_train.astype("category")))

num_factors = 2
num_features = 2
num_units1 = 30

X = tf.placeholder(tf.float32, shape=[None, num_features])
w1 = tf.Variable(tf.truncated_normal([num_features, num_units1]))
b1 = tf.Variable(tf.zeros([num_units1]))
hidden1 = tf.nn.tanh(tf.matmul(X, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([num_units1, num_factors]))
b2 = tf.Variable(tf.zeros([num_factors]))
p = tf.nn.softmax(tf.matmul(hidden1, w2) + b2)

t = tf.placeholder(tf.float32, [None, num_factors])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=p), axis = -1)
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_accuracy2= []
test_accuracy2 = []

i = 0

for _ in range(10001):
    i += 1
    _, loss_val = sess.run([train_step, loss], 
                           feed_dict = {X: X_train, t: train_t})
    if i % 1000 == 0 or i == 1:
        train_acc = sess.run(accuracy, feed_dict = {X: X_train, t: train_t})
        train_accuracy2.append(train_acc)
        test_acc = sess.run(accuracy, feed_dict = {X: X_test, t: test_t})
        test_accuracy2.append(test_acc)
        print('step:%d,loss:%f,Train:%f,Test:%f' 
              %(i,loss_val,train_acc,test_acc))

# step:1,loss:0.745210,Train:0.489333,Test:0.528000
# step:1000,loss:0.380812,Train:0.974667,Test:0.980000
# step:2000,loss:0.351109,Train:0.976000,Test:0.984000
# step:3000,loss:0.343328,Train:0.976000,Test:0.984000
# step:4000,loss:0.339571,Train:0.978667,Test:0.984000
# step:5000,loss:0.337136,Train:0.981333,Test:0.984000
# step:6000,loss:0.335124,Train:0.982667,Test:0.980000
# step:7000,loss:0.333378,Train:0.984000,Test:0.980000
# step:8000,loss:0.331939,Train:0.984000,Test:0.980000
# step:9000,loss:0.330917,Train:0.984000,Test:0.980000
# step:10000,loss:0.330134,Train:0.984000,Test:0.984000

grid2 = []
for x2 in np.linspace(min(df["x2"])-0.1,max(df["x2"])+0.1, 500):
    for x1 in np.linspace(min(df["x1"])-0.1,max(df["x1"])+0.1, 500):
        grid2.append((x1, x2))
out_val2 = np.argmax(sess.run(p, feed_dict = {X: grid2}), axis = 1)
out_val2 = pd.get_dummies(out_val2)
out_val21= np.array(out_val2.iloc[:, 0:1]).reshape((500, 500))
out_val22= np.array(out_val2.iloc[:, 1:2]).reshape((500, 500))

fig = plt.figure(figsize = (6, 6))
subplot = fig.add_subplot(1, 1, 1)
min_val1 = min(X_test["x1"]) - 0.1
max_val1 = max(X_test["x1"]) + 0.1
min_val2 = min(X_test["x2"]) - 0.1
max_val2 = max(X_test["x2"]) + 0.1
subplot.set_ylim([min_val2, max_val2])
subplot.set_xlim([min_val1, max_val1])
subplot.scatter(X_test1["x1"], X_test1["x2"], marker = "x")
subplot.scatter(X_test2["x1"], X_test2["x2"], marker = "o")
subplot.set_title("Predict NN", fontsize = 20)
subplot.imshow(out_val21, origin = "lower", 
               extent = (min_val1, max_val1, min_val2, max_val2),
               cmap = plt.cm.gray_r, alpha = 0.3)
subplot.imshow(out_val22, origin = "lower", 
               extent = (min_val1, max_val1, min_val2, max_val2),
               cmap = plt.cm.gray_r, alpha = 0.1)
plt.savefig('3.png', dpi=300)
```

![](/assets/DMSVM-circle/3.png)

Neural network split data with smooth line.

## DMSVM

```python
test_t1 = np.array(pd.get_dummies(t_test.astype("category")))
train_t1= np.array(pd.get_dummies(t_train.astype("category")))
test_t2 = np.where(test_t1 == 0, 1, 0)
train_t2= np.where(train_t1== 0, 1, 0)

num_factors = 2
num_features = 2
num_units1 = 30

X = tf.placeholder(tf.float32, shape=[None, num_features])
w1 = tf.Variable(tf.truncated_normal([num_features, num_units1]))
b1 = tf.Variable(tf.zeros([num_units1]))
hidden1 = tf.nn.relu(tf.matmul(X, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([num_units1, num_factors]))
b2 = tf.Variable(tf.zeros([num_factors]))
C = tf.constant([0.01])
prob = tf.matmul(hidden1, w2) + b2

t1 = tf.placeholder(tf.float32, [None, num_factors])
t2 = tf.placeholder(tf.float32, [None, num_factors])
cost = tf.reduce_sum(tf.square(w1))
hinge1 = tf.reduce_sum(tf.multiply(prob, t1), axis = 1)
hinge2 = tf.multiply(prob, t2)
hinge2 = tf.boolean_mask(hinge2, tf.not_equal(t2, 0.), axis = 0)
hinge2 = tf.reshape(hinge2, [-1, num_factors-1])
hinge2 = tf.reduce_max(hinge2, axis = 1)
hinge = tf.maximum(1 - hinge1 + hinge2, 0)
loss = tf.add(cost, tf.multiply(C, tf.reduce_sum(tf.square(hinge))))
train_step = tf.train.AdamOptimizer().minimize(loss)

output = tf.argmax(prob, axis = 1)
correct_prediction = tf.equal(output, tf.argmax(t1, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_accuracy3= []
test_accuracy3 = []

i = 0

for _ in range(10001):
    i += 1
    _, loss_val = sess.run([train_step, loss], 
                           feed_dict={X:X_train,t1:train_t1,t2:train_t2})
    if i % 1000 == 0 or i == 1:
        train_acc = sess.run(accuracy, 
                             feed_dict={X:X_train,t1:train_t1,t2:train_t2})
        train_accuracy3.append(train_acc)
        test_acc = sess.run(accuracy, 
                            feed_dict={X:X_test,t1:test_t1,t2:test_t2})
        test_accuracy3.append(test_acc)
        print('step:%d,loss:%f,Train:%f,Test:%f' 
              %(i,loss_val,train_acc,test_acc))

# step:1,loss:74.264999,Train:0.504000,Test:0.484000
# step:1000,loss:9.930771,Train:0.970667,Test:0.988000
# step:2000,loss:3.433367,Train:0.974667,Test:0.984000
# step:3000,loss:2.377622,Train:0.974667,Test:0.984000
# step:4000,loss:1.980406,Train:0.976000,Test:0.980000
# step:5000,loss:1.721627,Train:0.976000,Test:0.980000
# step:6000,loss:1.524906,Train:0.976000,Test:0.980000
# step:7000,loss:1.370554,Train:0.976000,Test:0.980000
# step:8000,loss:1.252676,Train:0.974667,Test:0.980000
# step:9000,loss:1.156818,Train:0.974667,Test:0.980000
# step:10000,loss:1.077520,Train:0.974667,Test:0.980000

grid3 = []
for x2 in np.linspace(min(df["x2"])-0.1,max(df["x2"])+0.1, 500):
    for x1 in np.linspace(min(df["x1"])-0.1,max(df["x1"])+0.1, 500):
        grid3.append((x1, x2))
out_val3 = sess.run(output, feed_dict = {X: grid3})
out_val3 = pd.get_dummies(out_val3)
out_val31= np.array(out_val3.iloc[:, 0:1]).reshape((500, 500))
out_val32= np.array(out_val3.iloc[:, 1:2]).reshape((500, 500))

fig = plt.figure(figsize = (6, 6))
subplot = fig.add_subplot(1, 1, 1)
min_val1 = min(X_test["x1"]) - 0.1
max_val1 = max(X_test["x1"]) + 0.1
min_val2 = min(X_test["x2"]) - 0.1
max_val2 = max(X_test["x2"]) + 0.1
subplot.set_ylim([min_val2, max_val2])
subplot.set_xlim([min_val1, max_val1])
subplot.scatter(X_test1["x1"], X_test1["x2"], marker = "x")
subplot.scatter(X_test2["x1"], X_test2["x2"], marker = "o")
subplot.set_title("Predict DMSVM", fontsize = 20)
subplot.imshow(out_val31, origin = "lower", 
               extent = (min_val1, max_val1, min_val2, max_val2),
               cmap = plt.cm.gray_r, alpha = 0.3)
subplot.imshow(out_val32, origin = "lower", 
               extent = (min_val1, max_val1, min_val2, max_val2),
               cmap = plt.cm.gray_r, alpha = 0.1)
plt.savefig('4.png', dpi=300)
```

![](/assets/DMSVM-circle/4.png)

DMSVM split data with smooth and linear line.

```python
fig = plt.figure(figsize = (15, 5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
min_val1 = min(X_test["x1"]) - 0.1
max_val1 = max(X_test["x1"]) + 0.1
min_val2 = min(X_test["x2"]) - 0.1
max_val2 = max(X_test["x2"]) + 0.1
ax1.scatter(X_test1["x1"], X_test1["x2"], marker = "x")
ax1.scatter(X_test2["x1"], X_test2["x2"], marker = "o")
ax1.set_title("SVM", fontsize = 15)
ax1.imshow(out_val11, origin = "lower", 
           extent = (min_val1, max_val1, min_val2, max_val2),
           cmap = plt.cm.gray_r, alpha = 0.3)
ax1.imshow(out_val12, origin = "lower", 
           extent = (min_val1, max_val1, min_val2, max_val2),
           cmap = plt.cm.gray_r, alpha = 0.1)
ax2.scatter(X_test1["x1"], X_test1["x2"], marker = "x")
ax2.scatter(X_test2["x1"], X_test2["x2"], marker = "o")
ax2.set_title("NN", fontsize = 15)
ax2.imshow(out_val21, origin = "lower", 
           extent = (min_val1, max_val1, min_val2, max_val2),
           cmap = plt.cm.gray_r, alpha = 0.3)
ax2.imshow(out_val22, origin = "lower", 
           extent = (min_val1, max_val1, min_val2, max_val2),
           cmap = plt.cm.gray_r, alpha = 0.1)
ax3.scatter(X_test1["x1"], X_test1["x2"], marker = "x")
ax3.scatter(X_test2["x1"], X_test2["x2"], marker = "o")
ax3.imshow(out_val31, origin = "lower", 
           extent = (min_val1, max_val1, min_val2, max_val2),
           cmap = plt.cm.gray_r, alpha = 0.3)
ax3.imshow(out_val32, origin = "lower", 
           extent = (min_val1, max_val1, min_val2, max_val2),
           cmap = plt.cm.gray_r, alpha = 0.1)
ax3.set_title("DMSVM", fontsize = 15)
plt.savefig('5.png', dpi=300)
```

![](/assets/DMSVM-circle/5.png)

