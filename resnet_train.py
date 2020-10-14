import os
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.api._v2.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(22)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

from make_data import load_pokemon, normalize, denormalize
from resnet import ResNet


def preprocess(x, y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # RGBA
    x = tf.image.resize(x, [244, 244])

    x = tf.image.random_flip_left_right(x)
    # x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224, 224, 3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y

def main():
    batchsz = 8

# creat train db   一般训练的时候需要shuffle。其它是不需要的。
    images, labels, table = load_pokemon('E:\\py_file\\4kmeinvv\\', mode='train')
    db_train = tf.data.Dataset.from_tensor_slices((images, labels))  # 变成个Dataset对象。
    db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)  # map函数图片路径变为内容。
# crate validation db
    images2, labels2, table = load_pokemon('E:\\py_file\\4kmeinvv\\', mode='val')
    db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
    db_val = db_val.map(preprocess).batch(batchsz)
# create test db
    images3, labels3, table = load_pokemon('E:\\py_file\\4kmeinvv\\', mode='test')
    db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
    db_test = db_test.map(preprocess).batch(batchsz)


# 首先创建Resnet18
    resnet = ResNet(5)
    resnet.build(input_shape=(batchsz, 224, 224, 3))

    resnet.summary()
# monitor监听器, 连续5个验证准确率不增加，这个事情触发。
#     early_stopping = EarlyStopping(
#         monitor='val_accuracy',
#         min_delta=0.001,
#         patience=20
#
#     )

# 网络的装配。
    resnet.compile(optimizer=optimizers.Adam(lr=1e-4),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])


    save_path = "./checkpoint/imgacs.ckpt"
    if os.path.exists(save_path + '.index'):
        print('--------------加载中-------------------')
        resnet.load_weights(save_path)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
    #                                               save_weights_only=True,
    #                                               save_best_only=True)

    # output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
    # keras.callbacks.TensorBoard(logdir),  # 将logdir传入Tensorboard中
    cp_callback = [
        keras.callbacks.ModelCheckpoint(save_path,  # 创建文件名
                                        save_best_only=True),  # save—best-only 默认保存最好的模型
        keras.callbacks.EarlyStopping(patience=20, min_delta=0.001)
        ]

# 完成标准的train，val, test;
# 标准的逻辑必须通过db_val挑选模型的参数，就需要提供一个earlystopping技术，
    history = resnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=3,
            callbacks=cp_callback)   # 1个epoch验证1次。触发了这个事情，提前停止了。
    resnet.evaluate(db_test)



# model.compile(optimizers='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#               metrics=['sparse_categorical_accuracy'])

    return resnet


if __name__=='__main__':
    main()
