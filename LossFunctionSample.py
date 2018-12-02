import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

# 定义一个简单的神经网络向前传播过程，这里就是简单的加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
# 自定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                               (y - y_) * loss_more,
                               (y_ - y) * loss_less))

# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成训练集
rdm = RandomState(1)
dataset_szie = 128
X = rdm.rand(dataset_szie, 2)

# 生成样本数据，y=x1+x2，加上一个随机量为了加入不可预测的噪音 -0.05~0.05
Y = [[x1 + x2 + rdm.rand() / 10 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_opt = tf.global_variables_initializer()
    sess.run(init_opt)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_szie
        end = min(start + batch_size, dataset_szie)
        sess.run(train_step,
                 feed_dict={x: X[start: end], y_: Y[start: end]})
    print(sess.run(w1))
