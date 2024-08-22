import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0 

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def forwardpropagation(inputs, layers):
    current_output = inputs

    for layer in layers:
        weighted_sum = np.dot(layer['weights'], current_output) + layer['bias']
        current_output = (relu(weighted_sum))
        
    return current_output

def input_layer(num_inputs, num_neurons):
    input_size = num_inputs
    weights = [[round(np.random.uniform(-1.0, 1.0), 2) for _ in range(input_size)] for _ in range(num_neurons)]
    bias = [round(np.random.uniform(-1.0, 1.0), 2) for _ in range(num_neurons)]
    return {"weights": weights, "bias": bias}
    
def hidden_layers(num_layers, num_neurons_per_layer, num_inputs):
    layers = []
    input_size = num_inputs

    for _ in range(num_layers):
        weights = np.random.uniform(-1.0, 1.0, (num_neurons_per_layer, input_size))
        bias = np.random.uniform(-1.0, 1.0, (num_neurons_per_layer))
        layers.append({"weights": weights, "bias": bias})
        input_size = num_neurons_per_layer

    return layers
    
def output_layer(num_inputs, hidden_layer_output, num_outputs=10):
    input_size = num_inputs
    weights = np.random.uniform(-1.0, 1.0, (num_outputs, input_size))
    bias = np.random.uniform(-1.0, 1.0, num_outputs)
    weighted_sum = np.dot(weights, hidden_layer_output) + bias
    output = softmax(weighted_sum)

    return output

num_inputs = 784
num_hidden_layers = 2
num_neurons_per_layer = 128
num_outputs = 10

input_layer_config = input_layer(num_inputs, num_neurons_per_layer)
hidden_layers_config = hidden_layers(num_hidden_layers, num_neurons_per_layer, num_inputs)
output_layer_config = {"weights": np.random.uniform(-1.0, 1.0, (num_outputs, num_neurons_per_layer)),
                       "bias": np.random.uniform(-1.0, 1.0, num_outputs)}

inputs = np.random.rand(num_inputs)
hidden_outputs = forwardpropagation(inputs, hidden_layers_config)
final_output = output_layer(num_neurons_per_layer, hidden_outputs, num_outputs)
print("Final Output:", final_output)