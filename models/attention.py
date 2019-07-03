from tensorflow.python.keras import layers


def multihead_attention_model(inputs):
    inputs_transposed = layers.Permute(dims=(2, 1))(inputs)
    # query_key = layers.Dot(axes=[1, 2])([inputs, inputs_transposed])
    query_key = layers.Dot(axes=2)([inputs, inputs])
    attentions = layers.Softmax(axis=-1)(query_key)  # TODO test it
    qkv = layers.Dot(axes=1)([attentions, inputs])
    return qkv
