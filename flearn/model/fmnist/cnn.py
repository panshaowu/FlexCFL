import os
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, context, value_and_grad
from mindspore.train import Accuracy

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers.pooling import Pooling2D


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.in_reshape = ops.Reshape()
        self.conv_layers = nn.SequentialCell(
            [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, pad_mode="same", has_bias=True, weight_init='XavierUniform'),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, pad_mode="same", has_bias=True, weight_init='XavierUniform'),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2)]
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(in_channels=64*7*7, out_channels=1024, activation="relu", weight_init='XavierUniform')
        self.dense2 = nn.Dense(in_channels=1024, out_channels=10, weight_init='XavierUniform')  # mindspore的nn.CrossEntropyLoss已包含softmax

    def construct(self, x):
        x = self.in_reshape(x, (-1, 1, 28, 28))
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense1(x)
        out = self.dense2(x)
        return out


class LossNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.softmax = nn.Softmax()
        self.loss_fn = nn.CrossEntropyLoss()

    def construct(self, data, label):
        logit = self.net(data)
        loss = self.loss_fn(logit, label)
        out = self.softmax(logit)
        return loss, out


class TrainNet(nn.Cell):
    """输入前向网络和loss函数, 执行一个batch的训练, 返回loss和acc

    Args:
        net (LossNet): 训练网络, 输入data和label, 输出loss和acc
        optim (Optimizer): 优化器对象.
    """
    def __init__(self, net, optim):
        super().__init__(auto_prefix=False)
        self.net = net
        self.optim = optim
        # loss_net的第2个输出不参与求导
        self.grad_net = value_and_grad(net, None, weights=optim.parameters, has_aux=True)
        self.depend = ops.Depend()

    def construct(self, data, label):
        (loss, pred), grads = self.grad_net(data, label)
        (loss, pred) = self.depend((loss, pred), self.optim(grads))
        return loss, pred


class ClientModel:
    def __init__(self, lr):
        self.class_num = 10
        self.net = Network()
        self.loss_net = LossNet(self.net)
        self.optim = nn.SGD(params=self.loss_net.trainable_params(), learning_rate=lr)
        self.train_net = TrainNet(self.loss_net, self.optim)
        self.metric = Accuracy('classification')
    
    def train_on_batch(self, data, label):
        loss, pred = self.train_net(data, label)
        self.metric.clear()
        self.metric.update(pred, label)
        acc = self.metric.eval()
        return loss, acc

    def train_on_epoch(self, dataset):
        iterator = dataset.create_tuple_iterator()
        loss_sum = .0
        self.metric.clear()
        for item in iterator:
            loss, pred = self.train_net(item[0], item[1])
            self.metric.update(pred, item[1])
            loss_sum += loss.asnumpy().tolist()
        loss = loss_sum / dataset.get_dataset_size()
        acc = self.metric.eval()
        return loss, acc

    def evaluate_on_batch(self, data, label):
        loss, pred = self.loss_net(data, label)
        self.metric.clear()
        self.metric.update(pred, label)
        acc = self.metric.eval()
        return loss, acc

    def evaluate(self, dataset):
        iterator = dataset.create_tuple_iterator()
        loss_sum = .0
        self.metric.clear()
        for item in iterator:
            loss, pred = self.loss_net(item[0], item[1])
            self.metric.update(pred, item[1])
            loss_sum += loss.asnumpy().tolist()
        loss = loss_sum / dataset.get_dataset_size()
        acc = self.metric.eval()
        return loss, acc


def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Reshape Layer
    model.add(Reshape((28, 28, 1)))

    # Conv1 Layer
    model.add(Conv2D(32, 5, activation='relu', padding='same'))
    # Pool1 Layer
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    # Conv2 Layer
    model.add(Conv2D(64, 5, activation='relu', padding='same'))
    # Pool2 Layer
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    # Flatten Layer
    model.add(Flatten())
    
    # Dense Layer
    model.add(Dense(1024, activation='relu'))
    # Output Layer
    model.add(Dense(10, 'softmax'))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model

def construct_model(trainer_type="fedavg", lr=0.003, platform="tf"):
    if platform == "tf":
        return _construct_client_model(lr)
    else:
        return ClientModel(lr)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # ms.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    client_model = ClientModel(0.01)
    x_np = np.random.rand(8, 784, )
    y_np = np.random.randint(10, size=(8,))
    x_ts = Tensor(x_np, dtype=ms.float32)
    y_ts = Tensor(y_np, dtype=ms.int32)

    params_iter0 = client_model.loss_net.parameters_and_names()
    parameter_dict0 = {key:value.copy() for key, value in params_iter0}

    for i in range(10):
        loss, acc = client_model.train_on_batch(x_ts, y_ts)
        print("#%d epoch, loss: %f, acc: %f" % (i, loss, acc))

    params_iter1 = client_model.loss_net.parameters_and_names()
    gradient = {}
    for name, param in params_iter1:
        tensor = param.asnumpy().astype(np.float32) - parameter_dict0[name].asnumpy()
        gradient[name] = ms.Parameter(tensor, name)

    res = .0
    for item in gradient.values():
        np_item = item.asnumpy()
        res += np.sum(np_item)

    print("test end, res %f" % res)
    ms.load_param_into_net(client_model.loss_net, parameter_dict0)
    loss, acc = client_model.evaluate_on_batch(x_ts, y_ts)
    print("retest load_param_into_net, loss: %f, acc: %f" % (loss, acc))
