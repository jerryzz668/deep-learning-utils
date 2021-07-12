import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import cv2

import matplotlib.pyplot as plt
import pywt
import pylab

# customer libraries
from utils import save_images, read_data
from metrics import SSIM, PSNR
from settings import *

from ops import bn, UPDATE_G_OPS_COLLECTION

class DerainNet:
    model_name = 'ReHEN'
    
    '''Derain Net: all the implemented layer are included (e.g. SEBlock,
                                                                HEU,
                                                                REU,
                                                                ReHEB).
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

        # 修改
        self.act = None
        self._eps = 1.1e-5  # epsilon
        self.res_scale = 1  # scaling factor of res block
        self.n_res_blocks = 2  # number of residual block
        
        # metrics
        self.ssim = SSIM(max_val=1.0)
        self.psnr = PSNR(max_val=1.0)

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

    # recurrent enhancement unit
    def REU(self, input_x, h, out_dim, scope='REU'):
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

        # channel attention block
        # se = self.SEBlock(h, out_dim, reduce_dim=int(out_dim/4))
        # h = self.leakyRelu(se)
        # return h, h
        h = self.residual_channel_attention_block(h, use_bn=True, name='scope')  # 修改
        return h, h

    # hierarchy enhancement unit
    def HEU1(self, input_x, is_training=False, scope='HEU'):
        with tf.variable_scope(scope) as scope:
            local_shortcut = input_x
            dense_shortcut = input_x
            
            for i in range(1, 3):
                with tf.variable_scope('ResBlock_{}'.format(i)):
                    with tf.variable_scope('Conv1'):
                        conv_tmp1 = slim.conv2d(local_shortcut, self.channel_dim,3,1)
                        conv_tmp1_bn = bn(conv_tmp1, is_training, UPDATE_G_OPS_COLLECTION)
                        out_tmp1 = tf.nn.relu(conv_tmp1_bn)

                    with tf.variable_scope('Conv2'):
                        conv_tmp2 = slim.conv2d(out_tmp1, self.channel_dim,3,1)
                        conv_tmp2_bn = bn(conv_tmp2, is_training, UPDATE_G_OPS_COLLECTION)
                        out_tmp2 = tf.nn.relu(conv_tmp2_bn)
                        conv_shortcut = tf.add(local_shortcut, out_tmp2)

                dense_shortcut = tf.concat([dense_shortcut, conv_shortcut], -1)
                local_shortcut = conv_shortcut

            with tf.variable_scope('Trans'):
                conv_tmp3 = slim.conv2d(dense_shortcut, self.channel_dim, 3,1)
                conv_tmp3_bn = bn(conv_tmp3, is_training, UPDATE_G_OPS_COLLECTION)
                conv_tmp3_se = self.SEBlock(conv_tmp3_bn, self.channel_dim, reduce_dim=int(self.channel_dim/4))
                out_tmp3 = tf.nn.relu(conv_tmp3_se)
                heu_f = tf.add(input_x, out_tmp3)

            return heu_f

    # def _residual_block(self, input_tensor, scope_name):
    def HEU(self, input_x, scope='HEU'):

        output = None
        with tf.variable_scope(scope):
            for i in range(2):
                if i == 0:
                    relu_1 = self.leakyRelu(slim.conv2d(input_x, 32, 3, 1, scope='block_{:d}_conv_1'.format(i)))
                    output = relu_1
                    input_x = output
                    # input_tensor = output  # 修改
                else:
                    relu_1 = self.leakyRelu(slim.conv2d(input_x, 32, 3, 1, scope='block_{:d}_conv_1'.format(i)))
                    relu_2 = self.leakyRelu(slim.conv2d(relu_1, 32, 3, 1, scope='block_{:d}_conv_2'.format(i)))

                    output = self.leakyRelu(tf.add(relu_2, input_x))
                    input_x = output

                    input_tensor1 = output  # 修改
            output = tf.add(output, input_tensor1)  # 修改
            # output = tf.add(output, input_tensor1)  # 修改
        return output

    # recurrent hierarchy enhancement block
    def ReHEB(self, input_x, h, is_training=False, scope='ReHEB'):
        with tf.variable_scope(scope):
            if input_x.get_shape().as_list()[-1] == 3:
                heu = input_x
            else:
                # heu = self.HEU(input_x, is_training=is_training)
                heu = self.HEU(input_x)
            reheb, h = self.REU(heu, h, out_dim=self.channel_dim)
        return reheb, h


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
            x = slim.conv2d(x, self.channel_dim//16, 3, 1)
            # x = self.act(x)

            x = slim.conv2d(x, self.channel_dim//16, 3, 1)
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

    def residual_group(self, x, use_bn, name):
        with tf.variable_scope("RG-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            for i in range(self.n_res_blocks):
                x = self.residual_channel_attention_block(x, use_bn, name=str(i))

            x = slim.conv2d(x, self.channel_dim, 3, 1)
            return x + skip_conn, x + skip_conn  # tf.math.add(x, skip_conn)

    def MAEB1(self, input_x, scope_name, dilated_factors=3):
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
            for i in range(1, dilated_factors + 1):
                d1 = self.leakyRelu(
                    slim.conv2d(input_x, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d1'))
                d2 = self.leakyRelu(slim.conv2d(d1, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d2'))
                dilate_c.append(d2)

            add = tf.add_n(dilate_c)
            # shape = add.get_shape().as_list()
            # output = self.SEBlock(add, shape[-1], reduce_dim=int(shape[-1] / 4))
            add = self.residual_channel_attention_block(add, use_bn=True, name='scope')  # 修改
            return add

    def dwt2d(self, x, wave='haar'):
        # shape x: (b, h, w, c)
        nc = int(x.shape.dims[3])
        # 小波波形
        w = pywt.Wavelet(wave)
        # 水平低频 垂直低频
        ll = np.outer(w.dec_lo, w.dec_lo)
        # 水平低频 垂直高频
        lh = np.outer(w.dec_hi, w.dec_lo)
        # 水平高频 垂直低频
        hl = np.outer(w.dec_lo, w.dec_hi)
        # 水平高频 垂直高频
        hh = np.outer(w.dec_hi, w.dec_hi)
        # 卷积核
        core = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
        core[:, :, 0, 0] = ll[::-1, ::-1]
        core[:, :, 0, 1] = lh[::-1, ::-1]
        core[:, :, 0, 2] = hl[::-1, ::-1]
        core[:, :, 0, 3] = hh[::-1, ::-1]
        core = core.astype(np.float32)
        kernel = np.array([core], dtype=np.float32)
        kernel = tf.convert_to_tensor(kernel)
        p = 2 * (len(w.dec_lo) // 2 - 1)
        with tf.variable_scope('dwt2d'):
            # padding odd length
            x = tf.pad(x, tf.constant([[0, 0], [p, p + 1], [p, p + 1], [0, 0]]))
            xh = tf.shape(x)[1] - tf.shape(x)[1] % 2
            xw = tf.shape(x)[2] - tf.shape(x)[2] % 2
            x = x[:, 0:xh, 0:xw, :]
            # convert to 3d data
            x3d = tf.expand_dims(x, 1)
            # 切开通道
            x3d = tf.split(x3d, int(x3d.shape.dims[4]), 4)
            # 贴到维度一
            x3d = tf.concat([a for a in x3d], 1)
            # 三维卷积
            y3d = tf.nn.conv3d(x3d, kernel, padding='VALID', strides=[1, 1, 2, 2, 1])
            # 切开维度一
            y = tf.split(y3d, int(y3d.shape.dims[1]), 1)
            # 贴到通道维
            y = tf.concat([a for a in y], 4)
            y = tf.reshape(y, (tf.shape(y)[0], tf.shape(y)[2], tf.shape(y)[3], 4 * nc))
            # 拼贴通道
            channels = tf.split(y, nc, 3)
            outputs = []
            for channel in channels:
                (cA, cH, cV, cD) = tf.split(channel, 4, 3)
                AH = tf.concat([cA, cH], axis=2)
                VD = tf.concat([cV, cD], axis=2)
                outputs.append(tf.concat([AH, VD], axis=1))
                pass
            outputs = tf.concat(outputs, axis=-1)
            pass
        return outputs

    def wavedec2d(self, x, level=1, wave='haar'):
        if level == 0:
            return x
        y = self.dwt2d(x, wave=wave)
        hcA = tf.floordiv(tf.shape(y)[1], 2)
        wcA = tf.floordiv(tf.shape(y)[2], 2)
        cA = y[:, 0:hcA, 0:wcA, :]
        cA = self.wavedec2d(cA, level=level - 1, wave=wave)
        cA = cA[:, 0:hcA, 0:wcA, :]
        hcA = tf.shape(cA)[1]
        wcA = tf.shape(cA)[2]
        cH = y[:, 0:hcA, wcA:, :]
        cV = y[:, hcA:, 0:wcA, :]
        cD = y[:, hcA:, wcA:, :]
        AH = tf.concat([cA, cH], axis=2)
        VD = tf.concat([cV, cD], axis=2)
        outputs = tf.concat([AH, VD], axis=1)
        return outputs

    def idwt2d(self, x, wave='haar'):
        # shape x: (b, h, w, c)
        nc = int(x.shape.dims[3])
        # 小波波形
        w = pywt.Wavelet(wave)
        # 水平低频 垂直低频
        ll = np.outer(w.dec_lo, w.dec_lo)
        # 水平低频 垂直高频
        lh = np.outer(w.dec_hi, w.dec_lo)
        # 水平高频 垂直低频
        hl = np.outer(w.dec_lo, w.dec_hi)
        # 水平高频 垂直高频
        hh = np.outer(w.dec_hi, w.dec_hi)
        # 卷积核
        core = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
        core[:, :, 0, 0] = ll[::-1, ::-1]
        core[:, :, 0, 1] = lh[::-1, ::-1]
        core[:, :, 0, 2] = hl[::-1, ::-1]
        core[:, :, 0, 3] = hh[::-1, ::-1]
        core = core.astype(np.float32)
        kernel = np.array([core], dtype=np.float32)
        kernel = tf.convert_to_tensor(kernel)
        s = 2 * (len(w.dec_lo) // 2 - 1)
        # 反变换
        with tf.variable_scope('idwt2d'):
            hcA = tf.floordiv(tf.shape(x)[1], 2)
            wcA = tf.floordiv(tf.shape(x)[2], 2)
            y = []
            for c in range(nc):
                channel = x[:, :, :, c]
                channel = tf.expand_dims(channel, -1)
                cA = channel[:, 0:hcA, 0:wcA, :]
                cH = channel[:, 0:hcA, wcA:, :]
                cV = channel[:, hcA:, 0:wcA, :]
                cD = channel[:, hcA:, wcA:, :]
                temp = tf.concat([cA, cH, cV, cD], axis=-1)
                y.append(temp)
                pass
            # nc * 4
            y = tf.concat(y, axis=-1)
            y3d = tf.expand_dims(y, 1)
            y3d = tf.split(y3d, nc, 4)
            y3d = tf.concat([a for a in y3d], 1)
            output_shape = [tf.shape(y)[0], tf.shape(y3d)[1],
                            2 * (tf.shape(y)[1] - 1) + np.shape(ll)[0],
                            2 * (tf.shape(y)[2] - 1) + np.shape(ll)[1], 1]
            x3d = tf.nn.conv3d_transpose(y3d, kernel, output_shape=output_shape, padding='VALID',
                                         strides=[1, 1, 2, 2, 1])
            outputs = tf.split(x3d, nc, 1)
            outputs = tf.concat([x for x in outputs], 4)
            outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[2], tf.shape(outputs)[3], nc))
            outputs = outputs[:, s:2 * (tf.shape(y)[1] - 1) + np.shape(ll)[0] - s, \
                      s:2 * (tf.shape(y)[2] - 1) + np.shape(ll)[1] - s, :]
            pass
        return outputs
