from easydict import EasyDict as edict
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class ConditionalDiscriminatorCifar10:
    
    def __init__(
            self,
            input_params: edict,
    ):
        self.img_height = input_params.img_height
        self.img_width = input_params.img_width
        self.num_channels = input_params.num_channels
        self.num_classes = input_params.num_classes
        self._model = self.create_model()
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs=inputs, **kwargs)
    
    @property
    def trainable_variables(self):
        return self._model.trainable_variables
    
    @property
    def model(self):
        return self._model
    
    def create_model(self):
        input_img = Input(shape=(self.img_height, self.img_width, self.num_channels))
        class_id = Input(shape=[1])
        
        embedded_id = layers.Embedding(
            input_dim=10,
            output_dim=50,
        )(class_id)
        
        embedded_id = layers.Dense(
            units=input_img.shape[1] * input_img.shape[2],
        )(embedded_id)
        
        embedded_id = layers.Flatten()(embedded_id)
        
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
        )(input_img)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2D(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2D(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Flatten()(x)
        
        x = layers.Concatenate()([x, embedded_id])
        x = layers.Dense(units=512, activation='relu')(x)
        
        x = layers.Dense(units=1)(x)
        
        model = Model(name='discriminator', inputs=[input_img, class_id], outputs=x)
        
        return model
