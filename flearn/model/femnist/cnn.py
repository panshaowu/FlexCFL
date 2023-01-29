import os
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.train import Model, Accuracy

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.in_reshape = ops.Reshape()
        self.conv_layers1 = nn.SequentialCell(
            [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, pad_mode="valid", has_bias=True),
             nn.ReLU(),
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, pad_mode="valid", has_bias=True),
             nn.ReLU(),
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, pad_mode="same", has_bias=True),
             nn.ReLU()]
        )
        self.dropout1 = nn.Dropout(keep_prob=0.4)
        self.conv_layers2 = nn.SequentialCell(
            [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, pad_mode="valid", has_bias=True),
             nn.ReLU(),
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, pad_mode="valid", has_bias=True),
             nn.ReLU(),
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, pad_mode="same", has_bias=True),
             nn.ReLU()]
        )
        self.dropout2 = nn.Dropout(keep_prob=0.4)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(in_channels=1024, out_channels=128, activation="relu")
        self.dropout3 = nn.Dropout(keep_prob=0.4)
        self.dense2 = nn.Dense(in_channels=128, out_channels=10, activation="softmax")

    def construct(self, x):
        x = self.in_reshape(x, (-1, 1, 28, 28))
        x = self.conv_layers1(x)
        x = self.dropout1(x)
        x = self.conv_layers2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        out = self.dense2(x)
        return out


class LossCell(nn.Cell):
    def __init__(self, net, l2_coef=0.001):
        super().__init__()
        self.net = net
        self.square = ops.Square()
        self.reduce_mean_false = ops.ReduceMean(keep_dims=False)
        self.l2_coef = l2_coef
        self.ce_loss = nn.CrossEntropyLoss()

    def construct(self, logit, label):
        l2_loss = self.l2_coef*(self.reduce_mean_false(self.square(self.net.dense1.weight)) \
            + self.reduce_mean_false(self.square(self.net.dense2.weight)))
        ce_loss = self.ce_loss(logit, label)
        return l2_loss + ce_loss


def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Reshape Layer
    model.add(Reshape((28, 28, 1)))

    # Conv1 Layer
    model.add(Conv2D(32, 3, activation='relu'))
    # BN1 Layer
    #model.add(BatchNormalization())
    # Conv2 Layer
    model.add(Conv2D(32, 3, activation='relu'))
    # BN2 Layer
    #model.add(BatchNormalization())
    # Conv3 Layer
    model.add(Conv2D(32, 5, strides=2, padding='same', activation='relu'))
    # BN3 Layer
    #model.add(BatchNormalization())
    # Droupout1
    model.add(Dropout(0.4))

    # Conv4 Layer
    model.add(Conv2D(64, 3, activation='relu'))
    # BN4 Layer
    #model.add(BatchNormalization())
    # Conv5 Layer
    model.add(Conv2D(64, 3, activation='relu'))
    # BN5 Layer
    #model.add(BatchNormalization())
    # Conv6 Layer
    model.add(Conv2D(64, 5, strides=2, padding='same', activation='relu'))
    # BN6 Layer
    #model.add(BatchNormalization())
    # Droupout2
    model.add(Dropout(0.4))

    # Flatten Layer
    model.add(Flatten())
    # Dense Layer
    model.add(Dense(128, 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # BN7 Layer
    #model.add(BatchNormalization())
    # Droupout3
    model.add(Dropout(0.4))
    # Output Layer
    model.add(Dense(10, 'softmax',  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model


def _construct_client_ms_model(lr):
    net = Network()
    loss_fn = LossCell(net)
    optim = nn.SGD(params=net.trainable_params(), learning_rate=lr)
    metric = Accuracy()
    return Model(net, loss_fn=loss_fn, optimizer=optim, metrics=metric)


def construct_model(trainer_type="fedavg", lr=0.003, platform="tf"):
    if platform == "tf":
        return _construct_client_model(lr)
    else:
        return _construct_client_ms_model(lr)


if __name__ == "__main__":
    os.environ["GLOG_v"] = "0"
    ms.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    x_np = np.random.rand(784,)
    x_ts = Tensor(x_np, dtype=ms.float32)
    net = Network()
    # 导出parameter_dict
    params_iter = net.parameters_and_names()
    parameter_dict = {name: param for name, param in params_iter}
    print("网络参数: ", parameter_dict)
    # 导入parameter_dict
    ms.load_param_into_net(net, parameter_dict)
    out = net(x_ts)
    print("推理结果: ", out)
