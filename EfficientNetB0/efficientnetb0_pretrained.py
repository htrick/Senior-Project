from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Reshape

class EfficientNetB0_Pretrained:
    def __init__(self, shape, num_outputs):
        self.shape = shape
        self.num_outputs = num_outputs
    
    def build(self):
        base_model = EfficientNetB0(input_tensor=Input(shape=self.shape), include_top=False, weights='imagenet', pooling='avg')
        base_model.trainable = False

        x = base_model.output

        x = Dense(200, activation='relu')(x)

        x = Dense(self.num_outputs, activation='linear')(x)
        x = Reshape((self.num_outputs,))(x)

        model = Model(inputs=base_model.inputs, outputs=x)
        return model

if __name__ == '__main__':
    m = EfficientNetB0_Pretrained(shape=(360,640,3), num_outputs=128)
    model = m.build()
