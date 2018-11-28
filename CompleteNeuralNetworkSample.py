# 完整神经网络样例程序

import tensorflow as tf
# NumPy 是一个科学计算的工具包，这里通过NumPy工具包生成模拟数据集
from numpy.random import RandomState

# 定义训练集 batch 大小
batch_size = 16

# 定义神经网络参数 w，stddev：标准差 seed：随机种子
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义输入变量
# 在 shape 的一个维度上使用 None 可以方便使用不同的 batch 大小。在训练时需要把数据分成比较小的 batch，
# 但是在测试时，可以一次性使用全部的数据。当数据集比较小时方便测试，但是数据集比较大时可能导致内存溢出
x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")

# 定义神经网络向前传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
# 使用 sigmoid 函数将 y 转换成 0~1 之间的数值。转换后 y 代表预测是正样本的概率，1-y 代表预测是负样本的概率
y = tf.sigmoid(y)
# 定义损失函数来刻画预测值与真实值的差距
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)
# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 2048
X = rdm.rand(dataset_size, 2)
# 定义规则来给出样本的标签。在这里 x1+x2<1 的样例被认为是正样本，其他是负样本。0 表示负样本，1 表示正样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行 TensorFlow 程序
with tf.Session() as sess:
    init_opt = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_opt)

    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练的轮数
    STEPS = 10000
    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本数据训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))






# 向前传播算法
'''
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# x = tf.placeholder(tf.float32, shape=[1, 2], name="input")
x = tf.placeholder(tf.float32, shape=[3, 2], name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as session:
    # 初始化操作
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    # print(session.run(y))
    # print(session.run(y, feed_dict={x: [[0.7, 0.9]]}))
    print(session.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
'''
