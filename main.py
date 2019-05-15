import os
from keras import layers, models, optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def show_enhanced_image(datagen, image_dir):
    """
    :函数功能: 显示数据增强后的图片
    :param datagen: 数据生成器
    :param image_dir: 图片文件夹
    :return:
    """
    fnames = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    img_path = fnames[2]
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()


def bulid_model():
    """
    构建网络：使用Conv2D和MaxPooling2D层交叠构成。
    Flatten层将3D输出展平到1D
    二分类问题最终使用sigmod激活
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(),
        metrics=['acc']
    )

    return model


def create_generator(dir):
    """
    创建数据生成器，这个生成器的作用是，将JPEG解码为RGB像素网格，然后将这些像素网格转换为浮点数向量，
    然后将像素值(0~255范围内)缩放到[0,1]区间。
    :param dir: 数据所在的目录
    :return: 返回一个生成器
    """
    dir_datagen = ImageDataGenerator(rescale=1. / 255)  # 将所有图像乘以1/255缩放
    generator = dir_datagen.flow_from_directory(
        dir,
        target_size=(128, 128),  # 图片大小调整为128 * 128
        batch_size=64,
        class_mode='binary',
        interpolation='lanczos',
    )  # 使用二进制
    return generator


def show_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """ 创建训练和验证集的目录 """
    train_dir = './pics/train/'
    validation_dir = './pics/val/'
    train_Parasitized_dir = os.path.join(train_dir, 'Parasitized')
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        channel_shift_range=50,
        zoom_range=0.05,
        rotation_range=90,
        shear_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
    )

    """ 打印增强的图片 """
    show_enhanced_image(datagen, train_Parasitized_dir)

    """ 构建网络 """
    model = bulid_model()

    """ 实例化训练生成器和验证生成器 """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        channel_shift_range=50,
        zoom_range=0.05,
        rotation_range=90,
        shear_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary',
        interpolation='lanczos',
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary',
        interpolation='lanczos',
    )

    """ 配置回调函数 """
    tensorboad = TensorBoard()
    checkpoint = ModelCheckpoint(
        filepath='dropout_Adam_lanczos.h5',
        save_best_only='True',
    )
    reduce_lr = ReduceLROnPlateau(
        factor=0.5,
        verbose=1,
        min_lr=0.0001,
    )

    """ 使用数据增强的方式，训练网络 """
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=240,
        epochs=50,
        verbose=2,
        callbacks=[tensorboad, checkpoint, reduce_lr],
        validation_data=validation_generator,
        validation_steps=100,
    )

    """ 显示训练结果 """
    show_results(history)
