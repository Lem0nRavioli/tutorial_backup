""" Deep dreams p303 """

import silence_tensorflow.auto
from tensorflow.keras.applications import inception_v3
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import scipy.misc
import imageio

tf.compat.v1.disable_eager_execution()

K.set_learning_phase(0)  # disable all training operation
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
# model.summary()
layer_contributions = {'mixed2': .2,
                       'mixed3': 3.,
                       'mixed4': 2.,
                       'mixed5': 1.5}
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.)
for layer_name in layer_contributions:
    print(layer_name)
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    print(activation)
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling


dream = model.input
grads = K.gradients(loss, dream)[0]  # compute the gradient of the dream in regard with loss
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # normalize gradient
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)  # retrieve value of the loss/gradients given input image


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at ', i, ': ', loss_value)
        x += step * grad_values
    return x


def preprocess_image(img_path):
    """ process images for inceptionV3 shape """
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    """ Undo the process made for inceptionV3 model """
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose(1, 2, 0)
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))

    x /= 2.
    x += .5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    # scipy.misc.imsave(fname, pil_img)
    imageio.imwrite(fname, pil_img)
    

step = .01
num_octave = 5
octave_scale = 1.4
iterations = 20
max_loss = 10.
base_image_path = 'testpic.jpg'
img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes = successive_shapes[::-1]  # reverse list of shape (increasing order)
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Preprocessing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)

    # reinjecting original picture data
    upscaled_shrunk_original = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream at scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')

# p310