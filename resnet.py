import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)


class ResnetBlock(keras.Model):
    def __init__(self, channels, strides=1):   #strides 卷积移动的步长
        super(ResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides

        self.conv1 = layers.Conv2D(channels, 3, strides=strides,
                                      padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = layers.Conv2D(channels, 3, strides=1,
                                   padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
        self.bn2 = keras.layers.BatchNormalization()


        if strides != 1:
            self.down_conv = layers.Conv2D(channels, 1, strides=strides,
                                           padding='valid')
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs

        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.bn2(x, training=training)

        #残差连接
        if self.strides != 1:
            residual = self.down_conv(inputs)
            residual = tf.nn.relu(residual)
            residual = self.down_bn(residual, training=training)

        x = x + residual
        x = tf.nn.relu(x)

        return x

    #Resnet18的实现
class ResNet(keras.Model):
    def __init__(self, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.stem = layers.Conv2D(initial_filters, 3, strides=3, padding='valid')

        #一共包含8个ResnetBlock模块，根连接的1层 + 输出的1层。
        #就是分成了4组，每组的第一个完成了高和宽的降维；第二个strides=1就是维度保持不变。

        self.blocks = keras.models.Sequential([
            ResnetBlock(initial_filters * 2, strides=3),
            ResnetBlock(initial_filters * 2, strides=1),

            ResnetBlock(initial_filters * 4, strides=3),
            ResnetBlock(initial_filters * 4, strides=1),

            ResnetBlock(initial_filters * 8, strides=2),
            ResnetBlock(initial_filters * 8, strides=1),

            ResnetBlock(initial_filters * 16, strides=2),
            ResnetBlock(initial_filters * 16, strides=1),
        ])

        self.final_bn = layers.BatchNormalization()
        self.avg_pool = layers.GlobalMaxPool2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        out = self.stem(inputs)
        out = tf.nn.relu(out)

        out = self.blocks(out, training=training)

        out = self.final_bn(out, training=training)

        out = self.avg_pool(out)

        out = self.fc(out)

        return out

def main():
    num_classes = 5

    resnet18 = ResNet(5)
    resnet18.build(input_shape=(4, 224, 224, 3))
    resnet18.summary()

if __name__ == '__main__':
    main()
