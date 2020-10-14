from __future__ import absolute_import, division, print_function
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow.keras.layers as layers
import time

from IPython import display
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig, axs = plt.subplots(4, 4)
    for i, ax in enumerate(axs.flat):
        ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    gen_loss_board(gen_loss)
    disc_loss_board(disc_loss)

    return gen_loss, disc_loss

def train(dataset, epochs):
    cnt = 0

    for epoch in range(epochs):
        batch_step = 0
        for image_batch in dataset:
            start = time.time()
            gen_loss, disc_loss = train_step(image_batch)
            cnt += 1

            if cnt > 0 and cnt % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('G_loss', gen_loss_board.result(), step=cnt)
                    tf.summary.scalar('D_loss', disc_loss_board.result(), step=cnt)

            if cnt > 0 and cnt % 50 == 0:
                print(
                    'epoch: {} / {}, batch:{} / {}, gen_loss{}, disc_loss {}, Time for batch process is {} sec'.format(
                        epoch + 1, epochs,
                        batch_step + 1, (train_images.shape[0] // BATCH_SIZE),
                        gen_loss, disc_loss,
                        time.time() - start))

            batch_step += 1
        generator_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        gen_loss_board.reset_states()
        disc_loss_board.reset_states()




EPOCHS = 150
noise_dim = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 256
num_examples_to_generate = 16

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# print(train_images.shape)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# print(train_dataset, train_labels)
generator = make_generator_model()
tf.keras.utils.plot_model(generator, to_file='generator.png', show_shapes=True, show_layer_names=True, expand_nested=True)
discriminator = make_discriminator_model()
tf.keras.utils.plot_model(discriminator, to_file='discriminator.png', show_shapes=True, show_layer_names=True, expand_nested=True)

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

gen_loss_board = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
disc_loss_board = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer, generator=generator,
                                 discriminator=discriminator)

train_log_dir = 'logs'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
