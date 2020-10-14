import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets
import os, glob, random, csv, time

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

def load_csv(root, filename, name2label):
    """加载CSV文件
    root：数据集根目录
    filename：CSV文件名
    name2label：类别名编码表
    """
    #判断.csv文件是否已经存在！
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        # print(len(images), images)

        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]

                writer.writerow([img, label])
            # print('written into csv file:', filename)
    images, labels = [], []
    with open(os.path.join(root, filename), mode='r', encoding='gbk') as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels

def load_pokemon(root, mode='train'):
    #创建数字编码表，范围0-4；
    name2label = {} #‘sq..’：0 类别名：类标签； 字典
    aaa = os.path.join(root) #列出所有目录；
    bbb = os.listdir(aaa)
    ccc = sorted(bbb)
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys()) #给每个类别编码一个数字
#读取label信息，保存索引文件images.csv
#【file1，file2，】，对应的标签【3,1】 2个一一对应的list对象 ***当前文档里面创建1,2,3,4...的文件夹放置不同类型的图片
#根据目录，把每个照片的路径提取出来，以及每个照片路径所对应的类别都存储起来，存储到csv

    images, labels = load_csv(root, 'images33.csv', name2label)
    if mode == 'train':
        images = images[:int(0.7 * len(images))]
        labels = labels[:int(0.7 * len(labels))]
    elif mode == 'val':
        images = images[int(0.7 * len(images)):int(0.85 * len(images))]
        labels = labels[int(0.7 * len(labels)):int(0.85 * len(labels))]
    else:
        images = images[int(0.85 * len(images)):]
        labels = labels[int(0.85 * len(labels)):]

    return images, labels, name2label


img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])

def normalize(x, mean=img_mean, std=img_std):
    """数据normalize
    上面两个值 均值和方差，统计所有imgennet的图片的均值和方差；
    所有者2个数据比较有意义，因为本质上所有图片的分布都和imagenet图片的分布基本一致；
    这6个数据基本是通用的，网上一搜就能查到。"""

    x = (x - mean)/std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    x = x * std + mean
    return x

def preprocess(x, y):
    print(x)
    x = tf.io.read_file(x) #图片路径
    x = tf.image.decode_jpeg(x, channels=3)  #转化图片对象，解码 如：1024 1024 3
    x = tf.image.resize(x, [64, 64]) #统一大小
    x = tf.image.random_flip_left_right(x)  # 随机的做一个左和右的翻转。
    # x = tf.image.random_crop(x, [224, 224, 3]) # 图片裁剪，这里注意这里裁剪到224*224，所以resize不能是224，比如250,250不然什么也没做。

    # x: [0,255]=> 0~1 或者-0.5~0.5   其次：normalizaion
    x = tf.cast(x, dtype=tf.float32) / 255.   #把值压缩到0-1之间
    x = normalize(x)

    y = tf.convert_to_tensor(y)
    print(x, y)

    return x, y

#root = 'E:/4kmeinv/'
#mode = 'train'




def main():
    images, labels, table = load_pokemon(r'E:\py_file\2266', 'train')
    # print('images', len(images), images)
    print('labels', len(labels), labels)

#转换成dataset类的对象，
    db = tf.data.Dataset.from_tensor_slices((images, labels))
    db = db.shuffle(1000).map(preprocess).batch(32)
    writter = tf.summary.create_file_writer('log')
    print(db)
    for step, (x, y) in enumerate(db):
        with writter.as_default():
            tf.summary.image('img', x, step=step, max_outputs=3)
            time.sleep(5)

if __name__ == '__main__':
    main()
