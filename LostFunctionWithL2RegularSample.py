import tensorflow as tf

# 获取一层神经网络边上的权重，并将这个权重的 L2 正则化损失加入名称为 losses 的集合中
def get_weight(shape, lambd):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

batch_size = 8

# 定义每层网络中的节点数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环生成 5 层全连接的神经网络
for i in range(1, n_layers):
    # layer_dimension[i] 为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层的权重的变量，并将这个变量的 L2 正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    # 定义偏移量
    bias = tf.Variable(tf.constant(0.1, shape=out_dimension))
    # 使用 ReLU 激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新到当前层的节点个数
    in_dimension = layer_dimension[i]

# 在定义神经网络前向传播的同时已经将所有的 L2 正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection 返回一个列表，这里将返回损失函数的不同部分，将他们加起来就可以得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))
