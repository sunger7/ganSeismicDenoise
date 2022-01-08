# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from model import *
import datetime
import time
import os
import numpy as np
import glob

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

@tf.function
def train_step(Generator,Discriminator,input_image, target, step,params):
    summary_writer,generator_optimizer,discriminator_optimizer = params
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = Generator(input_image)

        disc_real_output = Discriminator([input_image, target])
        disc_generated_output = Discriminator([input_image, gen_output])

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              Generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   Discriminator.trainable_variables)


        generator_optimizer.apply_gradients(zip(generator_gradients,
                                              Generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  Discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('disc_loss', disc_loss, step=step)

def generate_images(dataset):
    for batch in dataset:
        img_orig_batch = []
        img_target_batch = []
        for i in batch:
            orig,target = load(i.numpy().decode('UTF-8'))
            orig = tf.cast(orig, tf.float32)
            target = tf.cast(target, tf.float32)
            paddings = tf.constant([[0, 6], [0, 6],[0,0]])
            orig = tf.pad(orig, paddings, "CONSTANT")
            target = tf.pad(target, paddings, "CONSTANT")
            img_orig_batch.append(orig)
            img_target_batch.append(target)
        yield tf.convert_to_tensor(np.array(img_orig_batch)) ,tf.convert_to_tensor(np.array(img_target_batch))

def fit(Generator,Discriminator,train_ds, test_ds, steps,checkpoint,checkpoint_prefix,params):
    start = time.time()
    step = 0
    for (input_image, target) in train_ds:
        if (step) % 100 == 0:
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            #generate_images(Generator, example_input, example_target)
            print(f"Step: {step//100}00")

        train_step(Generator,Discriminator,input_image, target, step,params)

        # Training step
        if (step+1) % 100 == 0:
          print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 1000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        step = step + 1
def load(npy_file):
    origPath = "dataset/orig/"
    targetPath = "dataset/target/"
    #npy_file.decode
    # Read and decode an image file to a uint8 tensor
    input_image = np.load(origPath+npy_file)
    real_image = np.load(targetPath+npy_file)
    return np.expand_dims(input_image, axis=2), np.expand_dims(real_image, axis=2)
def load_image_train(image_file):
    return image_file
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fileList = []
    for i in glob.glob('dataset/orig/*.npy'):
        fileList.append(i.split('\\')[1])
    #print(fileList)
    #tf.enable_eager_execution()
    #tf.config.run_functions_eagerly(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    log_dir = "logs/"
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    discriminator = Discriminator()
    generator = Generator()
    generator.summary()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    train_dataset = tf.data.Dataset.from_tensor_slices(fileList[:-100])
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    BUFFER_SIZE = 200
    BATCH_SIZE = 16
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_gene = generate_images(train_dataset)
    try:
        test_dataset = tf.data.Dataset.from_tensor_slices(fileList[-100:])
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(fileList)
    test_dataset = test_dataset.map(load_image_train)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_gene = generate_images(test_dataset)
    params = summary_writer,generator_optimizer,discriminator_optimizer
    fit(generator,discriminator,train_gene,test_gene,100,checkpoint,checkpoint_prefix,params)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
