import numpy as np
import tensorflow as tf



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

##############TRAINING AND VALIDATION##############

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0 

batch_size = 32
epoch_size = 10

for epochs in range(epoch_size):
    loss = 0

    for i in range(0, len(train_images), batch_size) # Iterando sobre o dataset começando do 0, até len(train_images), em grupos de batch_size (32)
    batch_images = train_images[i:i+batch_size].reshape(batch_size, -1)
    batch_labels = train_labels[i:i+batch_size]

    hidden_outputs = forwardpropagation(batch_images.T, hidden_layers_config) # Forward propagation nas camadas ocultas
    
    logits = np.dot(output_layer_config["weights"], hidden_outputs) + output_layer_config["bias"] # Forward propagation na camada de output
    predictions = softmax(logits)

    batch_loss = cross_loss(predictions, batch_labels) # Calculo de perda do batch
    loss += batch_loss

    grad_output = softmax_grad(predictions, batch_labels) # Backpropagation do gradiente de output
    grad_weights_output = np.dot(grad_output, hidden_outputs.T)
    grad_bias_output = np.sum(grad_output, axis=1, keepdims=True) # O bias é simplesmente a soma dos outputs do gradiente, axis=1 indica que estamos somando ao longo do eixo dos exemplos do batch, keepdims=True mantém a dimensão original do array 2D 

    # Atualização dos pesos e bias
    output_layer_config["weights"], output_layer_config["bias"] = update_weights(
        output_layer_config["weights"], output_layer_config["bias"], grad_weights_output, grad_bias_output, learning_rate=0.01
    )

    
