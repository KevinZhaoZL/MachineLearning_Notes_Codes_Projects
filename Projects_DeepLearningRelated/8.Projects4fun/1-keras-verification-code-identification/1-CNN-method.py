# -*-coding:utf-8-*-
import time
from PIL import Image
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
import numpy as np

import random
import string

# # 生成验证码 captcha
# def generat(num, length):
#     characters = string.digits + string.ascii_uppercase
#     width, height, n_len, n_class = 170, 80, length, len(characters)
#     generator = ImageCaptcha(width=width, height=height)
#     for i in range(num):
#         random_str = ''.join([random.choice(characters) for j in range(n_len)])
#         img = generator.generate_image(random_str)
#         image = np.asarray(img)
#         image = image + np.random.randint(100,150, size=list(image.shape))
#         image = Image.fromarray(np.uint8(image))
#         image.save('.../verification-code/%s.jpg' % random_str)
# generat(2,5)

# 读取数据
import os

data_dir = '.../verification-code/'
test_dir = data_dir + 'test/'
train_dir = data_dir + 'train/'
IM_WIDTH = 170  # 图片宽度
IM_HEIGHT = 80  # 图片高度
characters = string.digits + string.ascii_uppercase

# 获得训练集图片张数
train_num = 0
for filename in os.listdir(train_dir):
    train_num = train_num + 1
train_imgs = np.zeros((train_num, IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
train_labels = [np.zeros((train_num, 36), dtype=np.uint8) for i in
                range(4)]
i = 0
for filename in os.listdir(train_dir):
    train_imgs[i] = np.array(Image.open(train_dir + filename))
    tmp_label = filename[:-4]
    j = 0
    for ch in tmp_label:
        train_labels[j][i, characters.find(ch)] = 1
        j = j + 1
    i = i + 1

test_num = 0
for filename in os.listdir(test_dir):
    test_num = test_num + 1
test_imgs = np.zeros((test_num, IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
test_labels = [np.zeros((test_num, 36), dtype=np.uint8) for i in
               range(4)]
i = 0
for filename in os.listdir(test_dir):
    test_imgs[i] = np.array(Image.open(test_dir + filename))
    tmp_label = filename[:-4]
    j = 0
    for ch in tmp_label:
        test_labels[j][i, characters.find(ch)] = 1
        j = j + 1
    i = i + 1

from keras.models import Model, Input
from keras import optimizers
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

input_tensor = Input((80, 170, 3))
x = input_tensor
for i in range(4):
    x = Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(36, activation='softmax', name='c%d' % (i + 1))(x)
     for i in range(4)]
model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.adadelta(),
              metrics=['accuracy'])
# 输出模型图片
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='CNN-model.png', show_shapes=True, show_layer_names=False)

num_epochs = 10
for epoch in range(num_epochs):
    start = time.time()
    history = model.fit(train_imgs, train_labels, batch_size=256, shuffle=True, verbose=0)
    score = model.evaluate(test_imgs, test_labels, verbose=0)
    loss = history.history['loss']
    train_acc = history.history['c1_acc'][0] * history.history['c2_acc'][0] * \
                history.history['c3_acc'][0] * history.history['c4_acc'][0]
    test_acc = score[5] * score[6] * score[7] * score[8]

    print "Epoch ", epoch + 1, \
        "  ,train_loss %.3f" % loss[0], "  ,train_acc %.4f" % train_acc, \
        "  ,test_loss %.3f" % score[0], "  ,test_acc %.4f" % test_acc, \
        ', time %.3f' % (time.time() - start), ' sec'
# 保存模型
model.save('CNN-model.h5')

# # 预测
# import string
# from keras.models import load_model
#
# characters = string.digits + string.ascii_uppercase
# model = load_model('CNN-model.h5')
#
#
# def decode(y):
#     y = np.argmax(np.array(y), axis=2)[:, 0]
#     return ''.join([characters[x] for x in y])
#
#
# pic = '.../pred/WTNI.jpg'
# pred_img = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
# pic_np = np.array(Image.open(pic))
# pred_img[0] = pic_np
# pic_pred = model.predict(pred_img)
# plt.title('real: %s\npred:%s' % (pic[-9:-4], decode(pic_pred)))
# plt.imshow(pic_np, cmap='gray')
# plt.show()
# plt.imshow()
