from gans.models import model


class CustomModel(model.Model):

    def __init__(
            self,
            fn,
    ):
        self.fn = fn
        super().__init__()

    def define_model(self):
        return self.fn()
