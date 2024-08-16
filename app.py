import numpy as np

def input_layer(num_inputs, num_neurons):
    num_inputs = 3
    
# Aqui, o peso da camada é um número aleatório entre -1 e 1 com até 2 casas decimais para cada input (no caso, 3), aplicado para cada perceptron da camada. Então, cada perceptron da camada receberá um total de *num_inputs* pesos.

    weights = [[round(np.random.uniform(-1.0, 1.0), 2) for _ in range(num_inputs)] for _ in range(num_neurons)]

# O bias (viés) é um número aleatório entre -1 e 1 com até 2 casas decimais para cada perceptron (neuron) da camada. 1 perceptron recebe 1 viés unico.

    bias = [round(np.random.uniform(-1.0, 1.0), 2) for _ in range(num_neurons)]

    return weights, bias
    

    
def hidden_layers(num_layers, num_inputs, num_neurons_per_layer):

# Definimos aqui que as camadas serão um array, isso porque podemos ter uma ou mais camadas, e cada uma delas possui N informações. 

    layers = []

    input_size = num_inputs

# Aqui, para cada layer existente, a função deve criar e inicializar os pesos e bias de cada camada. Então, junta-se essas camadas criadas no array "layers" e define que a quantidade de inputs que essa camada vai passar adiante será a quantidade de neurons da camada.

    for _ in range(num_layers):
        weights = np.random.uniform(-1.0, 1.0, (num_neurons_per_layer, input_size))
        bias = np.random.uniform(-1.0, 1.0, (num_neurons_per_layer))
        layers.append({"weights": weights, "bias": bias})
        input_size = num_neurons_per_layer

    return layers
        
    
    
# def output_layers():