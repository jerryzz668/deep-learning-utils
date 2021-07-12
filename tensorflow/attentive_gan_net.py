
import tensorflow as tf

from attentive_gan_model import cnn_basenet
from attentive_gan_model import vgg16


class GenerativeNet(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):

        super(GenerativeNet, self).__init__()
        self._vgg_extractor = vgg16.VGG16Encoder(phase='test')
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):

        return tf.equal(self._phase, self._train_phase)

    def _residual_block(self, input_tensor, name):

        output = None
        with tf.variable_scope(name):
            for i in range(5):
                if i == 0:
                    conv_1 = self.conv2d(inputdata=input_tensor,
                                         out_channel=32,
                                         kernel_size=3,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_relu_1'.format(i + 1))
                    output = relu_1
                    input_tensor = output
                else:
                    conv_1 = self.conv2d(inputdata=input_tensor,
                                         out_channel=32,
                                         kernel_size=1,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_conv_1'.format(i + 1))
                    conv_2 = self.conv2d(inputdata=relu_1,
                                         out_channel=32,
                                         kernel_size=1,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_2'.format(i))
                    relu_2 = self.lrelu(inputdata=conv_2, name='block_{:d}_conv_2'.format(i + 1))

                    output = self.lrelu(inputdata=tf.add(relu_2, input_tensor),
                                        name='block_{:d}_add'.format(i))
                    input_tensor = output

        return output

    def _conv_lstm(self, input_tensor, input_cell_state, name):

        with tf.variable_scope(name):
            conv_i = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_i')
            sigmoid_i = self.sigmoid(inputdata=conv_i, name='sigmoid_i')

            conv_f = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_f')
            sigmoid_f = self.sigmoid(inputdata=conv_f, name='sigmoid_f')

            cell_state = sigmoid_f * input_cell_state + \
                         sigmoid_i * tf.nn.tanh(self.conv2d(inputdata=input_tensor,
                                                            out_channel=32,
                                                            kernel_size=3,
                                                            padding='SAME',
                                                            stride=1,
                                                            use_bias=False,
                                                            name='conv_c'))
            conv_o = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_o')
            sigmoid_o = self.sigmoid(inputdata=conv_o, name='sigmoid_o')

            lstm_feats = sigmoid_o * tf.nn.tanh(cell_state)

            attention_map = self.conv2d(inputdata=lstm_feats, out_channel=1, kernel_size=3, padding='SAME',
                                        stride=1, use_bias=False, name='attention_map')
            attention_map = self.sigmoid(inputdata=attention_map)

            ret = {
                'attention_map': attention_map,
                'cell_state': cell_state,
                'lstm_feats': lstm_feats
            }

            return ret
