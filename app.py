import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0 

##############FUNCTIONS##############

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

def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2.0/shape[1])

def cross_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-8))

def softmax_grad(y_pred, y_true):
    return y_pred - y_true

def update_weights(weights, biases, grad_weights, grad_biases, learning_rate):
    weights -= learning_rate * grad_weights
    biases -= learning_rate * grad_biases
    return weights, biases

##############LAYERS##############


def input_layer(num_inputs, num_neurons):
    weights = he_initialization(num_inputs, num_neurons)
    bias = np.zeros(num_neurons)
    return {"weights": weights, "bias": bias}
    
def hidden_layers(num_layers, num_neurons_per_layer, num_inputs):
    layers = []

    for _ in range(num_layers):
        weights = he_initialization(num_inputs, num_neurons_per_layer)
        bias = np.zeros(num_neurons_per_layer)
        layers.append({"weights": weights, "bias": bias})
        input_size = num_neurons_per_layer

    return layers
    
def output_layer(num_inputs, hidden_layer_output, num_outputs=10):
    weights = he_initialization(num_inputs, num_outputs)
    bias = np.zeros(num_outputs)
    logits = np.dot(weights, hidden_layer_output) + bias
    output = softmax(logits)

    return output

num_inputs = 784
num_hidden_layers = 2
num_neurons_per_layer = 128
num_outputs = 10

input_layer_config = input_layer(num_inputs, num_neurons_per_layer)
hidden_layers_config = hidden_layers(num_hidden_layers, num_neurons_per_layer, num_inputs)
output_layer_config = {"weights": he_initialization((num_outputs, num_neurons_per_layer)),
                       "bias": np.zeros(num_outputs)}

inputs = train_images[0].flatten()
hidden_outputs = forwardpropagation(inputs, hidden_layers_config)
final_output = output_layer(num_neurons_per_layer, hidden_outputs, num_outputs)
print("Final Output:", final_output)