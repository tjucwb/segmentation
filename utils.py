import tensorflow as tf
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

"""input data"""
def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


"""download data"""
def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias,stride = 1,padding="SAME"):
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.bias_add(conv, bias)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2,padding="SAME"):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= stride
        output_shape[2] *= stride
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.bias_add(conv, b)


def conv2d(input,output_dim,k_h = 3,k_w =3,stride=1,padding='SAME',stddev=0.02,name='conv'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        init_w = tf.truncated_normal_initializer(stddev = stddev)
        init_b = tf.constant_initializer(0.0)
        W = tf.get_variable('_w',[k_h,k_w,input.get_shape()[-1],output_dim],initializer=init_w)
        b = tf.get_variable('_b',[output_dim],initializer=init_b)
    conv = tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding=padding)
    conv = tf.nn.bias_add(conv, b)
    return conv

def deconv2d(input,output_dim,output_shape=None,k_h = 3,k_w =3,stride=2,stddev=0.02,padding='SAME',name='deconv'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        init_w = tf.truncated_normal_initializer(stddev = stddev)
        init_b = tf.constant_initializer(0.0)
        W = tf.get_variable('W',[k_h,k_w,output_dim,input.get_shape()[-1]],initializer=init_w)
        b = tf.get_variable('b',[output_dim],initializer=init_b)
    if output_shape is None:
        output_shape = input.get_shape().as_list()
        output_shape[1] *= stride
        output_shape[2] *= stride
        output_shape[3] = output_dim
    deconv = tf.nn.conv2d_transpose(input, W, output_shape, strides=[1, stride, stride, 1], padding=padding)
    deconv = tf.nn.bias_add(deconv, b)
    return deconv

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


"""function to calculate balanced error rate"""
def b_error_rate(pre_label, test_pix_label):
    BER = []
    pre_label = np.squeeze(pre_label,axis=-1)
    ground_truth = test_pix_label
    ground_truth[:, :, :, 0] = np.zeros(shape=np.shape(ground_truth[:, :, :, 0]))
    ground_truth = ground_truth[:, :, :, 0] + ground_truth[:, :, :, 1] / ground_truth[:, :, :, 1].max()
    for i in range(len(ground_truth)):
        label = ground_truth[i]
        pre_l = pre_label[i]
        FP = np.sum((pre_l == 0)*(label > 0))/np.sum(label > 0)
        FN = np.sum((pre_l == 1)*(label == 0))/np.sum(label == 0)
        balanced_error_rate = 0.5*(FP + FN)
        BER.append(balanced_error_rate)
    return np.mean(np.array(BER))


"""compare the ground truth and the predicted segmentation, can be replaced by plot_segmentation_original"""
def plot_segmentatin(predict, ground_truth, number_of_showed_pictures):
    predict = np.squeeze(predict)
    ground_truth[:, :, :, 0] = np.zeros(shape=np.shape(ground_truth[:, :, :, 0]))
    ground_truth = ground_truth[:, :, :, 0] + ground_truth[:, :, :, 1] / ground_truth[:, :, :, 1].max()
    ground_truth = 45. * ground_truth / 255.
    showed_image = 105. / 255. * predict + ground_truth
    showed_image = np.expand_dims(showed_image, axis=-1)
    showed_image = np.concatenate((showed_image, showed_image, showed_image), axis=-1)
    #print('the shape of showed image is',np.shape(showed_image))
    for i in range(number_of_showed_pictures):
        plt.subplot(np.ceil(number_of_showed_pictures/4), 4, i+1)
        plt.imshow(showed_image[i, :, :, :], cmap='gray')


"""plot the gt and predicted segmentation on the original image"""
def plot_segmentation_original(predict, original_picture,ground_truth,number_of_pictures):
    img = 55/255./255.*original_picture
    temp = np.zeros(shape=np.shape(predict))
    overlay_pre = 55/255.*np.concatenate((temp,predict,temp),axis=-1)
    ground_truth[:, :, :, 0] = np.zeros(shape=np.shape(ground_truth[:, :, :, 0]))
    ground_truth = ground_truth[:, :, :, 0] + ground_truth[:, :, :, 1] / ground_truth[:, :, :, 1].max()
    ground_truth = np.expand_dims(ground_truth,axis = -1)
    overlay_gt = 55/255.*np.concatenate((temp,temp,ground_truth),axis=-1)
    img_show = img + overlay_pre + overlay_gt
    for i in range(number_of_pictures):
        plt.subplot(np.ceil(number_of_pictures/4), 4, i+1)
        plt.imshow(img_show[i, :, :, :])


"""show the learned features maps"""
def features_show(features,number_of_pictures):
    for i in range(number_of_pictures):
        plt.subplot(np.ceil(number_of_pictures/4), 4, i+1)
        show = np.concatenate((features[:, :, i],features[:, :, i],features[:, :, i]),axis=-1)
        plt.imshow(show, cmap='gray')



