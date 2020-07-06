import tensorflow as tf
import numpy as np
import imageio
from PIL import Image
from imagenet_classes import class_names


class VGG16(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.parameters = []

        # Conv1_1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv1)

        # Conv1_2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv2)

        # Pool1
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding='same'
        )

        # Drop1
        self.drop1 = tf.keras.layers.Dropout(0.2)

        # Conv2_1
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv3)

        # Conv2_2
        self.conv4 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv4)

        # Pool2
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding='same'
        )

        # Drop2
        self.drop2 = tf.keras.layers.Dropout(0.2)

        # Conv3_1
        self.conv5 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv5)

        # Conv3_2
        self.conv6 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv6)

        # Conv3_3
        self.conv7 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv7)

        # Pool3
        self.pool3 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding='same'
        )

        # Drop3
        self.drop3 = tf.keras.layers.Dropout(0.2)

        # Conv4_1
        self.conv8 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv8)

        # Conv4_2
        self.conv9 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv9)

        # Conv4_3
        self.conv10 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv10)

        # Pool4
        self.pool4 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding='same'
        )

        # Drop4
        self.drop4 = tf.keras.layers.Dropout(0.2)

        # Conv5_1
        self.conv11 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv11)

        # Conv5_2
        self.conv12 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv12)

        # Conv5_3
        self.conv13 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            dtype=tf.float32,
            activation=tf.nn.relu
        )
        self.parameters.append(self.conv13)

        # Pool5
        self.pool5 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding='same'
        )

        # Drop5
        self.drop5 = tf.keras.layers.Dropout(0.2)

        # FC
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=4096, dtype=tf.float32, activation=tf.nn.relu)
        self.drop6 = tf.keras.layers.Dropout(0.2)

        self.fc2 = tf.keras.layers.Dense(units=4096, dtype=tf.float32, activation=tf.nn.relu)
        self.drop7 = tf.keras.layers.Dropout(0.2)

        self.fc3 = self.dense2 = tf.keras.layers.Dense(units=1000, dtype=tf.float32, activation=tf.nn.softmax)

        self.parameters.append(self.fc1)
        self.parameters.append(self.fc2)
        self.parameters.append(self.fc3)

    def call(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.drop5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop6(x)
        x = self.fc2(x)
        x = self.drop7(x)
        output = self.fc3(x)
        return output

    # 读取预训练的权重数据
    def load_weights(self, weight_file):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i in range(len(self.parameters)):
            w_key_name = keys[2 * i]
            b_key_name = keys[2 * i + 1]
            print(2 * i, w_key_name, np.shape(weights[w_key_name]))
            print(2 * i + 1, b_key_name, np.shape(weights[b_key_name]))
            self.parameters[i].set_weights([weights[w_key_name], weights[b_key_name]])


if __name__ == '__main__':
    print(tf.__version__)

    # 图片传入并切割大小为224x224的格式
    img = imageio.imread('test4.jpg', pilmode='RGB')
    img_np = np.array(Image.fromarray(img).resize((224, 224)), dtype=np.float32)
    img_np = img_np[np.newaxis, :]

    # 预处理 从每个像素中减去在训练集上计算的平均RGB值
    mean = np.array([[[[123.68, 116.779, 103.939]]]], dtype=np.float32)
    img_np = img_np - mean

    # 创建初始化模型
    vgg_model = VGG16()
    vgg_model.build((None, 224, 224, 3))
    vgg_model.load_weights('vgg16_weights.npz')

    # 运行预测
    prob = vgg_model.predict(img_np)[0]
    results = np.argsort(-prob)[0:5]
    for r in results:
        print(class_names[r], prob[r])
