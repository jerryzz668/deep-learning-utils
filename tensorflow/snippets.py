import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import cv2
import matplotlib
from pylab import *

# customer libraries
from utils import save_images, read_data
from metrics import SSIM, PSNR
from settings import *


class DerainNet:
    model_name = 'ReMAEN'
    
    '''Derain Net: all the implemented layer are included (e.g. MAEB,
                                                                convGRU
                                                                shared channel attention,
                                                                channel attention).

        Params:
            config: the training configuration
            sess: runing session
    '''
    
    def __init__(self, config, sess=None):
        # config proto
        self.config = config
        self.channel_dim = self.config.channel_dim
        self.batch_size = self.config.batch_size
        self.patch_size = self.config.patch_size
        self.input_channels = self.config.input_channels
        
        # metrics
        self.ssim = SSIM(max_val=1.0)
        self.psnr = PSNR(max_val=1.0)

        # 修改
        self.act = None
        self._eps = 1.1e-5  # epsilon
        self.res_scale = 1  # scaling factor of res block
        self.n_res_blocks = 2  # number of residual block

        # create session
        self.sess = sess
    
    # global average pooling
    def globalAvgPool2D(self, input_x):
        global_avgpool2d = tf.contrib.keras.layers.GlobalAvgPool2D()
        return global_avgpool2d(input_x)
    
    # leaky relu
    def leakyRelu(self, input_x):
        leaky_relu = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)
        return leaky_relu(input_x)

    # squeeze-and-excitation block
    def SEBlock(self, input_x, input_dim=32, reduce_dim=8, scope='SEBlock'):
        with tf.variable_scope(scope) as scope:
            # global scale
            global_pl = self.globalAvgPool2D(input_x)
            reduce_fc1 = slim.fully_connected(global_pl, reduce_dim, activation_fn=tf.nn.relu)
            reduce_fc2 = slim.fully_connected(reduce_fc1, input_dim, activation_fn=None)
            g_scale = tf.nn.sigmoid(reduce_fc2)
            g_scale = tf.expand_dims(g_scale, axis=1)
            g_scale = tf.expand_dims(g_scale, axis=1)
            gs_input = input_x*g_scale
            return gs_input

    # GRU with convolutional version
    def convGRU(self, input_x, h, out_dim, scope='convGRU'):
        with tf.variable_scope(scope):
            if h is None:
                self.conv_xz = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xz')
                self.conv_xn = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xn')
                z = tf.nn.sigmoid(self.conv_xz)
                f = tf.nn.tanh(self.conv_xn)
                h = z*f
            else:
                self.conv_hz = slim.conv2d(h, out_dim, 3, 1, scope='conv_hz')
                self.conv_hr = slim.conv2d(h, out_dim, 3, 1, scope='conv_hr')

                self.conv_xz = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xz')
                self.conv_xr = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xr')
                self.conv_xn = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xn')
                r = tf.nn.sigmoid(self.conv_hr+self.conv_xr)
                z = tf.nn.sigmoid(self.conv_hz+self.conv_xz)
                
                self.conv_hn = slim.conv2d(r*h, out_dim, 3, 1, scope='conv_hn')
                n = tf.nn.tanh(self.conv_xn + self.conv_hn)
                h = (1-z)*h + z*n

        # shared channel attention block
        se = self.SEBlock(h, out_dim, reduce_dim=int(out_dim/4))
        h = self.leakyRelu(se)
        # h = self.residual_channel_attention_block(h, use_bn=True, name='scope')  # 修改
        return h, h
    def _conv_lstm(self, input_x, input_cell_state, name):

        with tf.variable_scope(name):
            self.conv_i = slim.conv2d(input_x, 32, 3, 1, scope='conv_i')
            sigmoid_i = tf.nn.sigmoid(self.conv_i,)
            self.conv_f = slim.conv2d(input_x, 32, 3, 1, scope='conv_f')
            sigmoid_f = tf.nn.sigmoid(self.conv_f)

            a = sigmoid_f * input_cell_state
            b = sigmoid_i * tf.nn.tanh(slim.conv2d(input_x, 32, 3, 1, scope='conv_c'))
            cell_state = tf.keras.layers.concatenate([a, b], axis=1)

            # cell_state = sigmoid_f * input_cell_state + \
            #             sigmoid_i * tf.nn.tanh(slim.conv2d(input_x, 32, 3, 1, scope='conv_c'))
            self.conv_o = slim.conv2d(input_x, 32, 3, 1, scope='conv_o')
            sigmoid_o = tf.nn.sigmoid(self.conv_o)

            lstm_feats = sigmoid_o * tf.nn.tanh(cell_state)


            return lstm_feats

    # def res_denseblock(self, input_tensor, scope_name):
    def res_denseblock(self, input_tensor, scope_name):
        with tf.variable_scope(scope_name):
            RDBs = []
            x_input = input_tensor

            """
            n_rdb = 5 ( RDB number )
            n_rdb_conv = 6 ( per RDB conv layer )
            """

            for k in range(3):
                if k == 0:
                    with tf.variable_scope('RDB_' + str(k)):
                        layers = []
                        layers.append(input_tensor)

                        # self.x = slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv'.format(k))
                        x = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv'.format(k)))

                        layers.append(x)

                        for i in range(1, 2):
                            x = tf.concat(layers, axis=-1)

                            # self.x = slim.conv2d(x, 32, 3, 1, scope = 'block_{:d}_conv_2'.format(i))
                            x = self.leakyRelu(slim.conv2d(x, 32, 3, 1, scope = 'block_{:d}_conv_2'.format(i)))

                            layers.append(x)
                else:
                    with tf.variable_scope('RDB_' + str(k)):
                        layers = []
                        layers.append(input_tensor)

                        x = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_3'.format(k)))

                        layers.append(x)

                        for i in range(1, 2):
                            x = tf.concat(layers, axis=-1)

                            x = self.leakyRelu(slim.conv2d(x, 32, 3, 1, scope='block_{:d}_conv_4'.format(i)))

                            layers.append(x)

                        # Local feature fusion
                        x = tf.concat(layers, axis=-1)
                        x = slim.conv2d(x, 32, 3, 1, scope='conv_last')

                        # Local residual learning
                        # x = input_tensor + x

                        RDBs.append(x)
                        input_tensor = x
            with tf.variable_scope('GFF_1x1'):
                x = tf.concat(RDBs, axis=-1)
                x = slim.conv2d(x, 32, kernel_size=1, stride=1, scope ='conv')

            with tf.variable_scope('GFF_3x3'):
                x = slim.conv2d(x, 32, kernel_size=3, stride=1, scope ='conv')

            # Global residual learning
            output = input_tensor + x

            return output

    # def _residual_block(self, input_tensor, scope_name):
    def MAEB(self, input_tensor, scope_name):

        output = None
        with tf.variable_scope(scope_name):
            for i in range(6):
                if i == 0:
                    relu_1 = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_1'.format(i)))
                    output = relu_1
                    input_tensor = output
                else:
                    relu_1 = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_1'.format(i)))
                    relu_2 = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_2'.format(i)))

                    output = self.leakyRelu(tf.add(relu_2, input_tensor))
                    input_tensor = output

        return output

    def adaptive_global_average_pool_2d(self, x):
        """
        In the paper, using gap which output size is 1, so i just gap func :)
        :param x: 4d-tensor, (batch_size, height, width, channel)
        :return: 4d-tensor, (batch_size, 1, 1, channel)
        """
        c = x.get_shape()[-1]
        return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

    def channel_attention(self, x, name):
        """
        Channel Attention (CA) Layer
        :param x: input layer
        :param f: conv2d filter size
        :param reduction: conv2d filter reduction rate
        :param name: scope name
        :return: output layer
        """
        with tf.variable_scope("CA-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = self.adaptive_global_average_pool_2d(x)
            # conv_tmp1 = slim.conv2d(local_shortcut, self.channel_dim, 3, 1)
            # x = slim.conv2d(x, f=f // reduction, k=1, name="conv2d-1")
            x = slim.conv2d(x, 32, 3, 1)
            # x = self.act(x)

            x = slim.conv2d(x, 32, 3, 1)
            x = tf.nn.sigmoid(x)
            return tf.multiply(skip_conn, x)

    def residual_channel_attention_block(self, x, use_bn, name):
        with tf.variable_scope("RCAB-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = slim.conv2d(x, self.channel_dim, 3, 1)
            x = tf.layers.BatchNormalization(epsilon=self._eps, name="bn-1")(x) if use_bn else x
            # x = self.act(x)

            x = slim.conv2d(x, self.channel_dim, 3, 1)
            x = tf.layers.BatchNormalization(epsilon=self._eps, name="bn-2")(x) if use_bn else x

            x = self.channel_attention(x, name="RCAB-%s" % name)
            return self.res_scale * x + skip_conn  # tf.math.add(self.res_scale * x, skip_conn)
    # multi-scale aggregation and enhancement block(MAEB)
    def skip(self, input_x, scope_name, dilated_factors=3):
        '''MAEB: multi-scale aggregation and enhancement block
            Params:
                input_x: input data
                scope_name: the scope name of the MAEB (customer definition)
                dilated_factor: the maximum number of dilated factors(default=3, range from 1 to 3)

            Return:
                return the output the MAEB
                
            Input shape:
                4D tensor with shape '(batch_size, height, width, channels)'
                
            Output shape:
                4D tensor with shape '(batch_size, height, width, channels)'
        '''
        dilate_c = []  
        with tf.variable_scope(scope_name):
            for i in range(1,dilated_factors+1):
                d1 = self.leakyRelu(slim.conv2d(input_x, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d1'))
                d2 = self.leakyRelu(slim.conv2d(d1, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d2'))
                dilate_c.append(d2)

            add = tf.add_n(dilate_c)
            # shape = add.get_shape().as_list()
            # output = self.SEBlock(add, shape[-1], reduce_dim=int(shape[-1]/4))
            # add = self.channel_attention(add, name='scope')  # 修改
            return add
