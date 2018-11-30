import math
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def matrix(i, j, random_fill=0):
    m = []
    for i in range(i):
        if random_fill:
            fill = rand(-0.2, 0.2)
        else:
            fill = 0.0
        m.append([fill] * j)
    return m


function_choice = 1


# activation functions
def act_fun(x):
    # sigmoid
    if function_choice == 0:
        return 1 / (1 + pow(math.e, -x))
    # tanh
    if function_choice == 1:
        return math.tanh(x)
    # linear
    if function_choice == 2:
        return x


# derivative of activation functions

def d_act_fun(y):
    if function_choice == 0:
        return y * (1 - y)
    if function_choice == 1:
        return 1 - y ** 2
    if function_choice == 2:
        return 1


# back propagation neuron network
# @input_num : input data dimension
# @output_num : output data dimension
# @hidden_layer_nums : dimensions of hidden layers

class BackPropagationNeuronNetwork:
    def __init__(self, input_num, hidden_layer_nums, output_num):
        self.input_num = input_num + 1
        self.hidden_layer_nums = hidden_layer_nums
        self.layers_num = len(hidden_layer_nums)
        self.output_num = output_num

        # create data matrices

        self.input_action = [1.0] * self.input_num
        self.hidden_layer_actions = []
        for layers in hidden_layer_nums:
            self.hidden_layer_actions.append([1.0] * layers)
        self.output_action = [1.0] * self.output_num

        # create weight matrices

        self.input_weight = matrix(self.input_num, self.hidden_layer_nums[0], 1)
        self.hidden_weights = [matrix(
            self.hidden_layer_nums[layer_num],
            self.hidden_layer_nums[layer_num + 1], 1)
            for layer_num in range(self.layers_num - 1)]
        self.output_weights = matrix(self.hidden_layer_nums[-1], self.output_num, 1)

        # create change matrices

        self.input_change = matrix(self.input_num, self.hidden_layer_nums[0])
        self.hidden_changes = [matrix(
            self.hidden_layer_nums[layer_num],
            self.hidden_layer_nums[layer_num + 1], 0)
            for layer_num in range(len(self.hidden_layer_nums) - 1)]
        self.output_change = matrix(self.hidden_layer_nums[-1], self.output_num)

        # create matrix for hidden layers error

        self.hidden_delta = [[0.0] * self.hidden_layer_nums[i] for i in range(self.layers_num)]

    def update(self, inputs):
        if len(inputs) != self.input_num - 1:
            raise ValueError('wrong number of inputs')

        # get inputs

        for i in range(self.input_num - 1):
            self.input_action[i] = inputs[i]

        # update the first hidden layer

        for j in range(self.hidden_layer_nums[0]):
            act = 0.0
            for i in range(self.input_num):
                act += self.input_action[i] * self.input_weight[i][j]
            self.hidden_layer_actions[0][j] = act_fun(act)

        # update hidden layers

        for k in range(self.layers_num - 1):
            for j in range(self.hidden_layer_nums[k + 1]):
                act = 0.0
                for i in range(self.hidden_layer_nums[k]):
                    act += self.hidden_layer_actions[k][i] * self.hidden_weights[k][i][j]
                self.hidden_layer_actions[k + 1][j] = act_fun(act)

        # update output layer

        for k in range(self.output_num):
            act = 0.0
            for j in range(self.hidden_layer_nums[-1]):
                act += self.hidden_layer_actions[-1][j] * self.output_weights[j][k]
            self.output_action[k] = act_fun(act)

        return self.output_action[:]

    def back_propagate(self, targets, n, m):
        if len(targets) != self.output_num:
            raise ValueError('wrong number of target values')

        # calculate output error

        output_deltas = [0.0] * self.output_num
        for k in range(self.output_num):
            error = targets[k] - self.output_action[k]
            output_deltas[k] = d_act_fun(self.output_action[k]) * error

        # calculate output error for last hidden layer

        for j in range(self.hidden_layer_nums[-1]):
            error = 0.0
            for k in range(self.output_num):
                error += output_deltas[k] * self.output_weights[j][k]
            self.hidden_delta[-1][j] = d_act_fun(self.hidden_layer_actions[-1][j]) * error

        # calculate output error for each hidden layer

        for i in range(self.layers_num - 1):
            for j in range(self.hidden_layer_nums[self.layers_num - i - 2]):
                error = 0.0
                for k in range(self.hidden_layer_nums[self.layers_num - i - 1]):
                    error += self.hidden_delta[self.layers_num - i - 1][k] \
                             * self.hidden_weights[self.layers_num - i - 2][j][k]
                self.hidden_delta[self.layers_num - i - 2][j] = \
                    d_act_fun(self.hidden_layer_actions[self.layers_num - i - 2][j]) * error

        # update output weights for output layer

        for j in range(self.hidden_layer_nums[-1]):
            for k in range(self.output_num):
                change = output_deltas[k] * self.hidden_layer_actions[-1][j]
                self.output_weights[j][k] += n * change + m * self.output_change[j][k]
                self.output_change[j][k] = change

        # update output weights for hidden layers

        for i in range(self.layers_num - 1):
            for j in range(self.hidden_layer_nums[i]):
                for k in range(self.hidden_layer_nums[i + 1]):
                    change = self.hidden_delta[i + 1][k] * self.hidden_layer_actions[i][j]
                    self.hidden_weights[i][j][k] += n * change + m * self.hidden_changes[i][j][k]
                    self.hidden_changes[i][j][k] = change

        # update input weights for input layer

        for i in range(self.input_num):
            for j in range(self.hidden_layer_nums[0]):
                change = self.hidden_delta[0][j] * self.input_action[i]
                self.input_weight[i][j] += n * change + m * self.input_change[i][j]
                self.input_change[i][j] = change

        # calculate error

        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.output_action[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print p[0][0] * 10, ' : ', self.update(p[0])

    def train(self, patterns, iterations=1000, n=0.5, m=0.1):
        # n: learning rate; m: momentum factor
        for i in range(iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                self.back_propagate(targets, n, m)


def main():
    pat = [
        [[0, 0, 0], [0, 0]],
        [[0, 0, 1], [1, 0]],
        [[0, 1, 0], [1, 0]],
        [[0, 1, 1], [0, 1]],
        [[1, 0, 0], [0, 1]],
        [[1, 0, 1], [0, 1]],
        [[1, 1, 0], [0, 1]],
        [[1, 1, 1], [1, 1]],
    ]

    n = BackPropagationNeuronNetwork(3, [8, 8], 2)
    for i in range(0, 100):
        n.train(pat)
    n.test(pat)


main()
