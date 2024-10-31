import json
from config import learning_rate, threshold, hidden_layer_size, stop_condition
import math
import random

w_i = []
b_i = []
w_j = []
b_j = []

def initialize():
    #note 1: hidden layer's size is 18.
    #w_i[] are weights between input layer and hidden layer.
    #w_j[] are weights between hidden layer and output layer.
    #b_i is bias of input layer.
    #b_j is bias of hidden layer.
    #each of 'w_i' and 'w_j' is a '3d' array and a '2d' array of weights respectively.
    global b_i
    global w_i
    global b_j
    global w_j
    for z in range(hidden_layer_size):
        b_i.append(random.uniform(-0.5, 0.5))

    for k in range(hidden_layer_size):
        w_i.append([])
        for i in range(5):
            w_i[k].append([])
            for j in range(5):
                w_i[k][i].append(random.uniform(-0.5, 0.5))
    for z in range(2):
        b_j.append(random.uniform(-0.5, 0.5))
    for k in range(2):
        w_j.append([])
        for i in range(hidden_layer_size):
            w_j[k].append(random.uniform(-0.5, 0.5))


def activation(Yin):
    return (2/(1+math.e**(-Yin)))-1

def activation_derivative(Yin):
    return 0.5*(1+activation(Yin))*(1-activation(Yin))

def encode_label(label):
    #target vector is [x, o]
    # [1, -1] for x and [-1, 1] for o
    label = label.lower()
    if label == 'x':
        return [1, -1]
    elif label == 'o':
        return [-1, 1]
    else:
        raise ValueError('please select a valid label among x or o') 

def decode_label(estimation):
    if estimation[0] > 0.5 and estimation[1] < 0.5:
        return 'X'
    elif estimation[0] < 0.5 and estimation[1] > 0.5:
        return 'O'
    else:
        return 'NONE'

def can_stop(loss):
    if loss < stop_condition:
        return True
    else:
        return False

def loss_func(output, target):
    return (output-target)**2

def max(num1, num2):
    if num1 >= num2:
        return num1
    else:
        return num2

def add_to_dataset(data, label):
    file = None
    try:
        file = open('dataset.txt', 'r')
        dataset = json.loads(file.readline())
        file.close()
        dataset.append({'features':data, 'label':label})
        file = open('dataset.txt', 'w')
        file.write(json.dumps(dataset))
        file.close()
    except:
        print('error occured')
    finally:
        if file:
            file.close()


def train():
    global b_i
    global b_j
    global w_i
    global w_j

    file = None
    
    try:

        initialize()

        file = open('dataset.txt', 'r')
        data_set = file.readline()
        file.close()

        data_set = json.loads(data_set)

        #run mlp algorithm:
        epochs = 0
        loss = 10000

        while not can_stop(loss):
            for data in data_set:
                
                hidden_layer_NI = []
                output_layer_NI = []
                hidden_layer_activations = []
                output_layer_activations = []
                delta_output = []
                delta_hidden = []

                #calc hidden layer activations:
                for k in range(hidden_layer_size):
                    Yin = 0
                    for i in range(5):
                        for j in range(5):
                            Yin += w_i[k][i][j]*data['features'][i][j]
                    Yin += b_i[k]
                    hidden_layer_NI.append(Yin)
                    hidden_layer_activations.append(activation(Yin))
                #calc output layer activations:
                for k in range(2):
                    Yin = 0
                    for i in range(hidden_layer_size):
                        Yin += w_j[k][i]*hidden_layer_activations[i]
                    Yin += b_j[k]
                    output_layer_NI.append(Yin)
                    output_layer_activations.append(activation(Yin))

                #calc delta_output:
                for k in range(2):
                    delta_output.append((encode_label(data['label'])[k]-output_layer_activations[k])*activation_derivative(output_layer_NI[k]))
                #calc delta_hidden:
                for j in range(hidden_layer_size):
                    D = 0
                    for k in range(2):
                        D += delta_output[k]*w_j[k][j]
                    delta_hidden.append(D*activation_derivative(hidden_layer_NI[j]))
                
                #update weights and bias:
                for i in range(hidden_layer_size):
                    b_i[i] += learning_rate*delta_hidden[i]
                for k in range(hidden_layer_size):
                    for i in range(5):
                        for j in range(5):
                            w_i[k][i][j] += learning_rate*delta_hidden[k]*data['features'][i][j]

                for i in range(2):
                    b_j[i] += learning_rate*delta_output[i]
                for k in range(2):
                    for i in range(hidden_layer_size):
                        w_j[k][i] += learning_rate*delta_output[k]*hidden_layer_activations[i]
            
            loss = max(
                loss_func(output_layer_activations[0], encode_label(data['label'])[0]),
                loss_func(output_layer_activations[1], encode_label(data['label'])[1])
            )
            epochs += 1
            #print(f"epoch: ", epochs)

        print(f'training successful through {epochs} epochs!')
        #print('weights:\n',)

    except ValueError as err:
        if err:
            print(err)
        else:
            print('An unExpected error occured!')

    finally:
        if file:
            file.close()


def test(test_data):
    try:
    
        hidden_layer_activations = []
        output_layer_activations = []

        #calc hidden layer activations:
        for k in range(hidden_layer_size):
            Yin = 0
            for i in range(5):
                for j in range(5):
                    Yin += w_i[k][i][j]*test_data[i][j]
            Yin += b_i[k]
            hidden_layer_activations.append(activation(Yin))
        #calc output layer activations:
        for k in range(2):
            Yin = 0
            for i in range(hidden_layer_size):
                Yin += w_j[k][i]*hidden_layer_activations[i]
            Yin += b_j[k]
            output_layer_activations.append(activation(Yin))

        return decode_label(output_layer_activations)


    except:
        print("An Unexpected error occured!")
        