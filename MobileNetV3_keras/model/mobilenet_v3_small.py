"""MobileNet v3 small models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from keras.layers import Flatten, Dense
from keras.utils.vis_utils import plot_model

from model.mobilenet_base import MobileNetBase

class MobileNetV3_Small(MobileNetBase):
    def __init__(self, shape, num_outputs, alpha=1.0, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if include classification layer.

        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Small, self).__init__(shape, num_outputs, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        #x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=200, s=1, squeeze=True, nl='HS')

        #x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=200, s=1, squeeze=True, nl='HS')

        #x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        x = self._conv_block(x, 200, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        #x = Reshape((1, 1, 576))(x)
        x = Reshape((1, 1, 200))(x)

        #x = Conv2D(1280, (1, 1), padding='same')(x)
        x = Conv2D(300, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            #x = Conv2D(128, (1, 1), padding='same', activation='linear')(x)
            #net = Conv2D(num_outputs, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
            x = Flatten()(x)
            x = Dense(self.num_outputs, activation='linear')(x)
            #x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        return model