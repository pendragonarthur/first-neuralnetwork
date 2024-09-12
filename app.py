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
    weights = he_initialization((num_inputs, num_neurons))
    bias = np.zeros(num_neurons)
    return {"weights": weights, "bias": bias}
    
def hidden_layers(num_layers, num_neurons_per_layer, num_inputs):
    layers = []

    for _ in range(num_layers):
        weights = he_initialization((num_inputs, num_neurons_per_layer))
        bias = np.zeros(num_neurons_per_layer)
        layers.append({"weights": weights, "bias": bias})
        num_inputs = num_neurons_per_layer

    return layers
    
def output_layer(num_inputs, hidden_layer_output, num_outputs=10):
    weights = he_initialization((num_inputs, num_outputs))
    bias = np.zeros(num_outputs)
    logits = np.dot(weights, hidden_layer_output) + bias
    output = softmax(logits)

    return output

num_inputs = 784 # Os inputs no codigo são imagens de 28x28 pixels, como estão transformadas em um unico array, cada imagem é um array de 784 elementos (1 elementos = 1 pixel) 
num_hidden_layers = 2
num_neurons_per_layer = 128
num_outputs = 10 # Como os unicos possiveis resultados são literalmente os numeros de 0 a 9, são 10 saídas possiveis no total

input_layer_config = input_layer(num_inputs, num_neurons_per_layer)
hidden_layers_config = hidden_layers(num_hidden_layers, num_neurons_per_layer, num_inputs)
output_layer_config = output_layer(num_neurons_per_layer, num_outputs)

##############TRAINING AND VALIDATION##############

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0 

batch_size = 32 # Boas práticas: minibatch = separar os dados em lotes (geralmente 32 ou 64 itens)
epoch_size = 10 

for epochs in range(epoch_size):
    loss = 0

    for i in range(0, len(train_images), batch_size): # Iterando sobre o dataset começando do 0, até len(train_images), em grupos de batch_size (32)
        batch_images = train_images[i:i+batch_size].reshape(batch_size, -1)
        batch_labels = train_labels[i:i+batch_size]

        hidden_outputs = forwardpropagation(batch_images, hidden_layers_config) # Forward propagation nas camadas ocultas
    

        # Forward propagation na camada de output
        logits = np.dot(output_layer_config["weights"], hidden_outputs) + output_layer_config["bias"] # Os logits são o produto escalar do peso da atual camada * a saída da camada anterior + o viés da camada
        predictions = softmax(logits)

        batch_loss = cross_loss(predictions, batch_labels) # Calculo de perda do batch
        loss += batch_loss

        grad_output = softmax_grad(predictions, batch_labels) # Backpropagation da camada de saída
        grad_weights_output = np.dot(grad_output, hidden_outputs.T) # .T serve para transpor a matriz do hidden_outputs. Se faz isso para que o calculo dao produto escalar requer que as dimensoes sejam compativeis
        grad_bias_output = np.sum(grad_output, axis=1, keepdims=True) # O bias é simplesmente a soma dos outputs do gradiente, axis=1 indica que estamos somando ao longo do eixo dos exemplos do batch, keepdims=True mantém a dimensão original do array 2D 

        # Atualização dos pesos e bias
        output_layer_config["weights"], output_layer_config["bias"] = update_weights(
        output_layer_config["weights"], output_layer_config["bias"], grad_weights_output, grad_bias_output, learning_rate=0.01)

        # Backpropagation das camadas ocultas

        grad_hidden = np.dot(output_layer_config["weights"].T, grad_output)
        for layer_index in reversed(range(len(hidden_layers_config))):
            current_layer = hidden_layers_config[layer_index]

            grad_relu = (hidden_outputs > 0).astype(float)
            grad_hidden = grad_hidden * grad_relu

            if layer_index == 0:
                grad_weights_hidden = np.dot(grad_hidden, batch_images)
            else:
                previous_outputs = hidden_layers_config[layer_index - 1]['outputs']
                grad_weights_hidden = np.dot(grad_hidden, previous_outputs.T)

            grad_bias_hidden = np.sum(grad_hidden, axis=1, keepdims=True)

            output_layer_config["weights"], output_layer_config["bias"] = update_weights(
            output_layer_config["weights"], output_layer_config["bias"], grad_weights_output, grad_bias_output, learning_rate=0.01)

            if layer_index > 1:
                grad_hidden = np.dot(current_layer['weights'].T, grad_hidden)

    print(f'Epoch: {epochs+1}/{epoch_size}, Loss: {loss/len(train_images)}')