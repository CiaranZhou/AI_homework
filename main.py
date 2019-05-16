from keras import layers, models, optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


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

    """ 构建网络 """
    model = bulid_model()

    """ 实例化训练生成器和验证生成器 """
    """
    创建数据生成器，这个生成器的作用是，将JPEG解码为RGB像素网格，然后将这些像素网格转换为浮点数向量，
    然后将像素值(0~255范围内)缩放到[0,1]区间。
    :param dir: 数据所在的目录
    :return: 返回一个生成器
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 将所有图像乘以1/255缩放
        rotation_range=270,  # 旋转角度范围
        shear_range=0.2,  # 剪切范围
        horizontal_flip=True,  # 水平翻转
        vertical_flip=True,  # 垂直翻转
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),  # 图片大小调整为128*128
        batch_size=64,
        class_mode='binary',  # 使用二进制
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
    tensorboad = TensorBoard()  # 可视化
    checkpoint = ModelCheckpoint(
        filepath='dropout_Adam_lanczos.h5',  # 模型文件名称
        save_best_only='True',  # 选择保留最好的模型,默认取val_loss最小的那个,patience取默认值10
    )
    reduce_lr = ReduceLROnPlateau(  # 设置当指定值不再变动时降低学习率,指定值取默认值val_loss
        factor=0.5,  # 降低为原学习率的0.5
        verbose=1,  # 降低学习率时输出日志
        min_lr=0.0001,  # 设置学习率下限
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
