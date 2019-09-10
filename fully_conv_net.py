from __future__ import print_function
import data_input as di
import tensorflow as tf
import numpy as np
import utils
import datetime
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '2', 'batch size for training')
tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
tf.flags.DEFINE_float('learning_rate', '1e-4', 'Learning rate for Adam Optimizer')
tf.flags.DEFINE_string('model_dir', './model', 'Path to vgg model mat')
tf.flags.DEFINE_bool('RESIZE_TAG','True','True/ False, to determine if the images are supposed to be resized')
tf.flags.DEFINE_integer('MAX_ITERATION', '100', 'total iterations')
tf.flags.DEFINE_integer('NUM_OF_CLASSESS', '2', 'number of segmented classes')
tf.flags.DEFINE_integer('IMAGE_SIZE', '224', 'image size after resizing')
tf.flags.DEFINE_float('Penalize_weight', '300', 'weights for the object pixels')

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = FLAGS.NUM_OF_CLASSESS
IMAGE_SIZE = FLAGS.IMAGE_SIZE


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def fine_tune_net(image, keep_prob):
    """
    the network to be fine tuned and used to perform the semantic segmentation
    :param image: input image.
    :param keep_prob: for doupout
    :return: annotation prediction, probability map and 2nd last layer of vgg
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = image - mean_pixel

    with tf.variable_scope("fine_tune"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]   # 14x14x512

        pool5 = utils.max_pool_2x2(conv_final_layer) # 7x7x512

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)  # 7x7x4096

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)  # 7x7x4096

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # upscale
        deconv_shape1 = image_net["pool4"].get_shape()  # 14x14x512
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))   #
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([4,4, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        #conv_t3 = tf.layers.conv2d_transpose(fuse_2,NUM_OF_CLASSESS,16,strides=(8,8),padding='SAME')
        #conv_t3.set_shape([None,IMAGE_SIZE,IMAGE_SIZE,NUM_OF_CLASSESS])

        conv_t3 = tf.nn.softmax(conv_t3,axis=-1)
        annotation_pred = tf.argmax(conv_t3, axis=-1, name="prediction")


    return tf.expand_dims(annotation_pred, axis=-1), conv_t3,conv_final_layer


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 2], name="annotation")

    pred_annotation, logits,features = fine_tune_net(image, keep_probability)
    #loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                     labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                      name="entropy")))
    loss = -tf.reduce_mean(tf.log(logits + 1e-6)*annotation)
    loss_summary = tf.summary.scalar("loss", loss)

    trainable_var = tf.trainable_variables()

    train_op = train(loss, trainable_var)

    print("Setting up dataset")
    image_options = {'resize_tag': FLAGS.RESIZE_TAG, 'resize_size': IMAGE_SIZE, 'weight': FLAGS.Penalize_weight}
    try:
        dataset = di.database(image_options)
        train_data, test_data, train_annotations, test_annotations = dataset.data_segmentation()
    except Exception as e:
        print(e)
        exit()

    #print('size of train data and annotations', np.shape(train_data), np.shape(train_annotations))
    #print('size of test data and annotations', np.shape(test_data), np.shape(test_annotations))
    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # the save the graph
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    iterations_train = []
    train_l = []
    iterations_test = []
    BBER = []
    # training
    for itr in range(FLAGS.MAX_ITERATION):
        train_images, train_annotation = dataset.fetch_batch(FLAGS.batch_size)
        feed_dict = {image: train_images, annotation: train_annotation, keep_probability: 0.65}

        sess.run(train_op, feed_dict=feed_dict)

        if itr % 5 == 0:
            train_loss, summary_str,lo = sess.run([loss, loss_summary,logits], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            train_writer.add_summary(summary_str, itr)
            iterations_train.append(itr)
            train_l.append(train_loss)

        if itr % 25 == 0:
            valid_images, valid_annotations = test_data, test_annotations
            valid_loss, summary_sva, predict = sess.run([loss, loss_summary, pred_annotation], feed_dict={image: valid_images, annotation: valid_annotations,
                                                   keep_probability: 1.0})
            print("%s Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
            print('the size of predict is',np.shape(predict))

            # add validation loss to TensorBoard
            validation_writer.add_summary(summary_sva, itr)
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            balanced_error_rate = utils.b_error_rate(predict,valid_annotations)
            print('the balanced error rate now is: ', balanced_error_rate)
            iterations_test.append(itr)
            BBER.append(balanced_error_rate)


    valid_images, valid_annotations = test_data, test_annotations
    predict,feature_map = sess.run([pred_annotation,features],feed_dict={image: valid_images, annotation: valid_annotations,keep_probability: 1.0})
    #print('the total number of bright points in', np.sum(predict>0))
    #print('the shape of feature map is',np.shape(feature_map))
    #plot_segmentatin(predict,valid_annotations,20)
    '''plot the segmentation'''

    print('visualization...')
    plt.figure(num='segmentation_result')
    utils.plot_segmentation_original(predict,valid_images,valid_annotations,20)

    '''plot the feature map'''
    plt.figure(num='features_map1')
    utils.features_show(feature_map[0], 20)
    plt.figure(num='features_map2')
    utils.features_show(feature_map[8], 20)
    plt.figure(num='features_map3')
    utils.features_show(feature_map[16], 20)

    '''plot the train loss'''
    plt.figure(num='train_loss')
    plt.plot(iterations_train,train_l)

    '''plot BER'''
    plt.figure(num='BER')
    plt.plot(iterations_test,BBER)
    plt.show()



if __name__ == "__main__":
    tf.app.run()