import data_input as di
import tensorflow as tf
from functools import partial
import datetime
import numpy as np
import matplotlib.pyplot as plt

'''parameter settings'''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '2', 'batch size for training')
tf.flags.DEFINE_integer('epoch', '25', 'epoch for training')
tf.flags.DEFINE_float('lr', '5e-5', 'Learning rate for Adam Optimizer')
tf.flags.DEFINE_bool('shuffle','True','True/ False, True: shuffle the dataset in every epoch')
tf.flags.DEFINE_integer('NUM_OF_CLASSESS', '2', 'batch size for training')
tf.flags.DEFINE_float('threshold_seg', '0.45', 'threshold for determining whether the pixel belongs to object or not')
tf.flags.DEFINE_float('weight', '250.0', 're-weight the pixels belonging to object for imbalance dataset')
tf.flags.DEFINE_float('reg_co', '5e-4', 'weight for regularization')

model_name = 'segmentation'
savedir = 'results/' + model_name + '-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
batch_size = FLAGS.batch_size

def autoencoder(input,is_training,name,class_num):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        '''encoder'''
        net = tf.layers.conv2d(input,32,2,padding='same',name='conv1')
        net = tf.layers.batch_normalization(net,training=is_training,name='bn1')
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net,32,2,padding='same',name='conv2')
        net_concat1 = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net_concat1,pool_size=2,strides=2,padding='same',name='maxpool1')
        net = tf.layers.conv2d(net,64,2,padding='same',name='conv3')
        net = tf.layers.batch_normalization(net,training=is_training,name='bn2')
        net_concat2 = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net_concat2,pool_size=2,strides=2,padding='same',name='maxpooling2')
        net = tf.layers.conv2d(net, 128, 2, padding='same',name='conv5')
        net = tf.layers.batch_normalization(net, training=is_training,name='bn3')
        net = tf.nn.relu(net)
        net_latent = net

        '''decoder'''
        net = tf.layers.conv2d_transpose(net,64,2,2,padding='same',name='deconv1')
        net = tf.concat([net,net_concat2],axis=-1)
        net = tf.layers.batch_normalization(net, training=is_training,name='bn4')
        net = tf.nn.relu(net)
        net = tf.layers.conv2d_transpose(net,32,2,2,padding='same',name='deconv2')
        net = tf.concat([net,net_concat1],axis=-1)
        net = tf.layers.batch_normalization(net,training=is_training,name='bn5')
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net,32,3,padding='same',name='conv6')
        net = tf.layers.batch_normalization(net, training=is_training,name='bn6')
        net = tf.nn.relu(net)
        out = tf.layers.conv2d(net, class_num, 1, padding='same', name='decision',activation=tf.nn.softmax)
        assert(out.get_shape().as_list()[-1] == class_num)
        assert(out.get_shape().as_list()[-2] == 1328)
        assert(out.get_shape().as_list()[-3] == 664)

        return out, net_latent

def main(argv=None):
    '''make the graph for tensorflow'''
    t_data = tf.placeholder(dtype=tf.float32, shape=[None, 664, 1328, 3])
    t_label = tf.placeholder(dtype=tf.float32,shape=[None, 664, 1328, 2])
    is_train = tf.placeholder(dtype=tf.bool)
    sess = tf.Session()

    net_cascade1 = partial(autoencoder, name='train_net1',class_num= FLAGS.NUM_OF_CLASSESS)
    #net_cascade2 = partial(autoencoder, name='train_net2',class_num= 2)

    pre_mask_final,la_space = net_cascade1(t_data,is_train)
    #pre_mask_final = net_cascade2(pre_mask_1)

    #loss
    loss = tf.reduce_mean(tf.log(pre_mask_final + 1e-10)*t_label)

    '''initialize the variables'''
    T_vars = tf.trainable_variables()
    loss = -loss + FLAGS.reg_co*tf.add_n([tf.nn.l2_loss(w) for w in T_vars])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(loss,var_list=T_vars)
    sess.run(tf.global_variables_initializer())

    '''training'''
    # import data
    print('inputting data.....')
    image_options = {'resize_tag': False, 'weight': FLAGS.weight}
    dataset = di.database(image_options)
    train_data, test_data, train_pix_label, test_pix_label = dataset.data_segmentation()

    # training step
    print('training begins.....')
    data_num = train_data.shape[0]
    batch_num = data_num // batch_size
    data_idx = np.arange(data_num)
    loss_value = []
    BBER = []
    for ep in range(FLAGS.epoch):
        for ind in range(batch_num):
            feed_data = train_data[ind * batch_size:(ind + 1) * batch_size]
            feed_label = train_pix_label[ind * batch_size:(ind + 1) * batch_size]
            train_op,los = sess.run([train_step,loss],feed_dict={t_data:feed_data,t_label:feed_label,is_train:True})
            loss_value.append(los)
        if FLAGS.shuffle:
            shuffle_index = data_idx
            np.random.shuffle(shuffle_index)
            train_data = train_data[shuffle_index]
            train_pix_label = train_pix_label[shuffle_index]
        pre = sess.run(pre_mask_final,feed_dict={t_data:test_data,t_label:test_pix_label,is_train:False})
        pre[:, :, :, 1][pre[:, :, :, 1] > FLAGS.threshold_seg] = 1
        pre = np.argmax(pre,axis=-1)
        BER = b_error_rate(pre,test_pix_label)
        BBER.append(BER)
        print('......')
        print('the sum of all labels',np.sum(pre))
        print('the BER is for epoch'+str(ep)+' '+'is ', BER)

    '''plot all needed pics'''
    print('visualization.....')
    pre_final,latent_space = sess.run([pre_mask_final,la_space],feed_dict={t_data:test_data,t_label:test_pix_label,is_train:False})
    pre_final[:, :, :, 1][pre_final[:, :, :, 1] > FLAGS.threshold_seg] = 1
    pre_final = np.argmax(pre_final,axis=-1)
    '''plot the learned features'''
    plt.figure(num='latent feature')
    for i in range(1,17):
        plt.subplot(8,2,i)
        plt.imshow(latent_space[0, :, :, i], cmap='gray')
    #print(np.sum(pre_final>0))

    '''plot the segmentation'''
    plt.figure(num='segmentation_result')
    segmentation_visualize(test_pix_label,test_data,pre_final)

    '''plot the loss'''
    plt.figure(num='loss')
    iteration = range(len(loss_value))
    plt.plot(iteration,loss_value)

    '''plot BER'''
    plt.figure(num='ber')
    epoch = range(len(BBER))
    plt.plot(epoch,BBER)
    plt.show()


"""function to calculate the balanced error rate"""
def b_error_rate(pre_label, test_pix_label):
    BBER = []
    ground_truth = test_pix_label
    ground_truth[:, :, :, 0] = np.zeros(shape=np.shape(ground_truth[:, :, :, 0]))
    ground_truth = ground_truth[:, :, :, 0] + ground_truth[:, :, :, 1] / ground_truth[:, :, :, 1].max()
    for i in range(len(ground_truth)):
        label = ground_truth[i]
        pre_l = pre_label[i]
        FP = np.sum((pre_l == 0)*(label > 0))/np.sum(label > 0)
        FN = np.sum((pre_l == 1)*(label == 0))/np.sum(label == 0)
        BER = 0.5*(FP + FN)
        BBER.append(BER)
    return np.mean(np.array(BBER))


"""function to plot the segmentation results"""
def segmentation_visualize(test_pix_label,test_data,pre_final):
    test_ground_truth = test_pix_label
    test_ground_truth[:, :, :, 0] = np.zeros(shape=np.shape(test_ground_truth[:, :, :, 0]))
    ground_truth = test_ground_truth[:, :, :, 0] + test_ground_truth[:, :, :, 1] / test_ground_truth[:, :, :, 1].max()
    #print(np.sum(ground_truth>0))
    ground_truth = np.expand_dims(ground_truth,axis=-1)
    image_show = 100/255./255.*test_data
    pre = np.expand_dims(pre_final,axis=-1)
    temp = np.zeros(shape=np.shape(pre))
    image_show = image_show + 100/255.*np.concatenate((pre,temp,temp),axis=-1) + 100/255.*np.concatenate((temp,ground_truth,temp),axis=-1)
    num_test_data = len(test_data)
    for i in range(num_test_data):
        plt.subplot(np.ceil(num_test_data / 4), 4, i + 1)
        plt.imshow(image_show[i, :, :, :], cmap='gray')

if __name__ == "__main__":
    tf.app.run()











