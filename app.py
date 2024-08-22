import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0 

def input_layer(num_inputs, num_neurons):
    input_size = num_inputs
    weights = [[round(np.random.uniform(-1.0, 1.0), 2) for _ in range(input_size)] for _ in range(num_neurons)]
    bias = [round(np.random.uniform(-1.0, 1.0), 2) for _ in range(num_neurons)]
    outputs = []

    for i in range(num_neurons):
        weighted_sum = sum(weights * inputs for weights, inputs in zip(weights[i], num_inputs)) + bias[i]
        outputs.append(relu(weighted_sum))

    return outputs
    

    
def hidden_layers(num_layers, num_neurons_per_layer, previous_layer_output):
    outputs = []
    layers = []
    input_size = previous_layer_output

    for _ in range(num_layers):
        weights = np.random.uniform(-1.0, 1.0, (num_neurons_per_layer, input_size))
        bias = np.random.uniform(-1.0, 1.0, (num_neurons_per_layer))
        layers.append({"weights": weights, "bias": bias})
        input_size = num_neurons_per_layer

    return layers
    
def output_layer(num_inputs, hidden_layer_output):

    input_size = num_inputs

    for _ in range(input_size):
        dot_product = np.dot(hidden_layer_output, outputlayer) + 