"""
This module provides custom TensorFlow Keras layers for encoding tabular data.

The module contains two main classes:
1. CategoricalEncodingLayer: For encoding categorical variables using embeddings.
2. NumericalEncodingLayer: For encoding numerical variables using separate dense layers.

These layers are designed to be used in neural network models that process tabular data,
particularly in the context of the SAINT paper.

Classes:
    CategoricalEncodingLayer: A layer for encoding categorical variables.
    NumericalEncodingLayer: A layer for encoding numerical variables.

Each layer takes a 2D input tensor and outputs a 3D tensor, where each feature
is encoded separately and then concatenated along the feature dimension.

Usage example:
    ```python
    import tensorflow as tf
    from tabular.encoding import CategoricalEncodingLayer, NumericalEncodingLayer

    # For categorical features
    cat_input = tf.keras.layers.Input(shape=(3,))
    cat_encoded = CategoricalEncodingLayer(category_sizes=[10, 20, 15], embedding_size=32)(cat_input)

    # For numerical features
    num_input = tf.keras.layers.Input(shape=(5,))
    num_encoded = NumericalEncodingLayer(layer_size=64)(num_input)

    # Combine the encodings
    combined = tf.keras.layers.Concatenate()([cat_encoded, num_encoded])
    ```

Note:
    These layers are particularly useful for processing tabular data in deep learning models,
    allowing for separate treatment of categorical and numerical features while maintaining
    a consistent output structure.
"""
import tensorflow as tf

class CategoricalEncodingLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer for encoding categorical variables using embeddings.

    This layer takes categorical input features and applies separate embedding layers
    to each feature. It then concatenates the embedded representations.

    Args:
        layer_size (int): The size of the layer (not used in current implementation).
        category_sizes (list of int): A list containing the number of unique categories
            for each input feature.
        embedding_size (int): The size of the embedding vector for each category.
        **kwargs: Additional keyword arguments to be passed to the parent Layer class.

    Attributes:
        layer_size (int): The size of the layer.
        category_sizes (list of int): The number of unique categories for each input feature.
        embedding_size (int): The size of the embedding vector for each category.
        k (int): The number of input features (set in the build method).
        embedding_layers (list of tf.keras.layers.Embedding): List of embedding layers,
            one for each input feature.

    Input shape:
        2D tensor with shape: (batch_size, k), where k is the number of input features.

    Output shape:
        3D tensor with shape: (batch_size, k, embedding_size).

    Raises:
        ValueError: If the number of category_sizes doesn't match the number of input features.

    Example:
        ```python
        layer = CategoricalEncodingLayer(layer_size=64, 
                                         category_sizes=[10, 20, 15], 
                                         embedding_size=32)
        input_tensor = tf.keras.Input(shape=(3,))
        output_tensor = layer(input_tensor)
        ```
    """
    def __init__(self, category_sizes, embedding_size, **kwargs):
        super(CategoricalEncodingLayer, self).__init__(**kwargs)
        self.category_sizes = category_sizes
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.k = input_shape[1]
        category_sizes = self.category_sizes
        embedding_size = self.embedding_size
        if len(category_sizes) != self.k:
            msg = f"""
            ERROR:
            input shape is `{input_shape}` but `{len(category_sizes)}` category_sizes are given.  
            """
            raise ValueError(msg)
            
        self.embedding_layers = [tf.keras.layers.Embedding(category_sizes[i], embedding_size) 
                             for i in range(self.k)]

    def call(self, inputs):
        # Reshape input to (batch_size, k, 1)
        reshaped_input = tf.expand_dims(inputs, axis=-1)
        
        # Apply embedding layers to each of the k dimensions
        outputs = [layer(reshaped_input[:, i:i+1]) for i, layer in enumerate(self.embedding_layers)]
        
        # Concatenate the outputs
        return tf.concat(outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, self.embedding_size)

    def get_config(self):
        config = super(CategoricalEncodingLayer, self).get_config()
        config.update({
            'category_sizes': self.category_sizes,
            'embedding_size': self.embedding_size
        })
        return config

import tensorflow as tf
class NumericalEncodingLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that applies separate dense layers to each dimension of the input.
    This is done as per the SAINT paper.
    
    This layer takes an input of shape (batch_size, k) and outputs a tensor of shape 
    (batch_size, k, layer_size). It creates k separate dense layers, each applied to 
    a single dimension of the input, and then concatenates the results.

    Args:
        layer_size (int): The number of units in each dense layer, which becomes
                          the size of the last dimension in the output.
        activation (str or callable, optional): Activation function to use for the dense layers.
                                                Defaults to 'relu'.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        2D tensor with shape: (batch_size, k)

    Output shape:
        3D tensor with shape: (batch_size, k, layer_size)

    Example:
        ```python
        input_layer = tf.keras.layers.Input(shape=(5,))
        custom_dense = NumericalEncodingLayer(layer_size=10)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=custom_dense)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
        ```

    Note:
        The number of separate dense layers (k) is automatically inferred from 
        the second dimension of the input shape during the build method.
    """
    def __init__(self, layer_size, activation='relu', **kwargs):
        super(NumericalEncodingLayer, self).__init__(**kwargs)
        self.layer_size = layer_size
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.k = input_shape[1]
        self.dense_layers = [tf.keras.layers.Dense(self.layer_size, activation=self.activation) 
                             for _ in range(self.k)]

    def call(self, inputs):
        # Reshape input to (batch_size, k, 1)
        reshaped_input = tf.expand_dims(inputs, axis=-1)
        
        # Apply dense layers to each of the k dimensions
        outputs = [layer(reshaped_input[:, i:i+1]) for i, layer in enumerate(self.dense_layers)]
        
        # Concatenate the outputs
        return tf.concat(outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, self.layer_size)

    def get_config(self):
        config = super(NumericalEncodingLayer, self).get_config()
        config.update({'layer_size': self.layer_size})
        return config
