from tensorflow.python import keras

from gans.models import model


class SequentialModel(model.Model):

    def __init__(self, layers):
        self._layers = layers
        super().__init__()

    @property
    def layers(self):
        return self._layers

    def define_model(self) -> keras.Model:
        inputs = self._layers[0]
        current_input = inputs
        for layer in self.layers[1:]:
            outputs = layer(current_input)
            current_input = outputs
        model = keras.Model(name=self.model_name, inputs=inputs, outputs=outputs)
        return model
