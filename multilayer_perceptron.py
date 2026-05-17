import numpy as np
import matplotlib.pyplot as plt

class Linear_Layer:
    def __init__(self, cln_amount, nln_amount, alpha, beta): #cln_amount - current layer neuron amount; nln_amount - next layer neuron amount;
        self.weights = np.random.rand(cln_amount, nln_amount)
        self.biases = np.random.rand(1, nln_amount)
        self.deriv_W = None #derivative w.r.t. weight matrix
        self.deriv_b = None #derivative w.r.t. biases vector
        self.alpha = alpha #learning rate
        self.beta = beta #SGD momentum
        self.pl_values = None #preceding layer values
        
    def forward(self, input):
        self.pl_values = input #save values incoming to the current layer from previous
        
        out_array = input @ self.weights + self.biases
        return out_array
    
    def backward(self, prior_deriv):
        r_nm = prior_deriv.shape[0] #row number in the output vector from a linear layer
        
        
        if r_nm == 0: #if batch size is zero make sure we don't calculate any derivative for the current layer
            return np.zeros(self.pl_values.shape) 
        
        #The layout for the current layer is (PxN) so from P neurons in this layer it will compute N neurons for the next layer
        A_prev = self.pl_values.T # (KxP)T = (PxK)
        curr_deriv_W = (A_prev @ prior_deriv) / r_nm # (PxK) * (KxN) = (PxN)
    
        W_t = self.weights.T
        g_next = prior_deriv @ W_t # (KxN) * (NxP) = (KxP) so the output of this layer will be the same as output of the previous activation layer's function
        curr_deriv_b = prior_deriv.sum(axis = 0) / r_nm 
        #since we have size of prior derivative matrix KxN and we have N amount of neurons for this layer we need to tally all of the rows in the prior_deriv matrix and divide by number of samples, hence K
        
        self.update_weights(curr_deriv_W)
        self.update_biases(curr_deriv_b)
        
        return g_next
    
    
    def update_weights(self, new_deriv_W):
        if(self.deriv_W is None): 
            self.deriv_W = new_deriv_W
        else:
            self.deriv_W = self.beta * self.deriv_W + (1 - self.beta) * new_deriv_W #we add previous derivative's values with current derivative's
    
        self.weights -= self.deriv_W * self.alpha
    
    def update_biases(self, new_deriv_b):
        if(self.deriv_b is None):
            self.deriv_b = new_deriv_b
        else:
            self.deriv_b = self.beta * self.deriv_b + (1 - self.beta) * new_deriv_b #we add previous derivative's values with current derivative's
        
        self.biases -= self.deriv_b * self.alpha

class Sigmoid:
    @staticmethod
    def derivative(x):
        deriv = x * (1 - x)
        return deriv
    
    def __init__(self, beta):
        self.t_values = None #values transformed by the current activation function
    
    def forward(self, input):
        t_values = 1 / (1 + np.exp(-1 * input)) #transformed input
        self.t_values = t_values
        
        return t_values
    
    def backward(self, prior_deriv):
        local_deriv = Sigmoid.derivative(self.t_values)
        #since size of prior_deriv matrix and input_matrix(passed into the activation function) is the same, matrices are multiplied elementwise
        g_next = local_deriv * prior_deriv

        return g_next
    
class HyperTan:
    @staticmethod
    def derivative(x):
        return 1 - (x)**2
    
    def __init__(self, beta):
        self.t_values = None #values transformed by the current activation function
    
    def forward(self, input):
        output = np.tanh(input)
        self.t_values = output
        
        return output
    
    def backward(self, prior_deriv):
        local_deriv = HyperTan.derivative(self.t_values)
        g_next = local_deriv * prior_deriv

        return g_next
    
class ReLu:
    @staticmethod
    def derivative(x):
        #basically a mask of incoming matrix is created; if a singular value in the values matrix is more than 0, then on the same position in the mask matrix there would be True, otherwise False.
        #thereafter all values would be converted to 1.0 or 0.0
        return (x > 0).astype(np.float32)
    
    def __init__(self, beta):
        self.pl_values = None #values fetched from the prior layer
    
    def forward(self, input):
        self.pl_values = input
        output = np.copy(input)
        output[output < 0] = 0
        
        return output
    
    def backward(self, prior_deriv):
        local_deriv = ReLu.derivative(self.pl_values)
        g_next = local_deriv * prior_deriv
        
        return g_next
    
class MSE:
    @staticmethod
    def derivative(y_real, y_predicted):
        return 2*(y_predicted - y_real)
    
    def __init__(self, beta):
        self.pl_values = None
        self.g_error = 0
    
    def forward(self, predicted_y, genuine_y):
        self.pl_values = predicted_y
        self.g_error = np.mean((predicted_y - genuine_y)**2)
        return predicted_y
    
    def backward(self, y_real, y_predicted):
        g_next = MSE.derivative(y_real, y_predicted)

        return g_next

class MLP:
    def __init__(self, alpha, beta, batch_size):
        self.l1 = Linear_Layer(2, 4, alpha, beta)
        self.l2 = Linear_Layer(4, 4, alpha, beta)
        self.l3 = Linear_Layer(4, 1, alpha, beta)
        self.a1 = HyperTan(beta)
        self.a2 = ReLu(beta)
        self.a3 = Sigmoid(beta)
        self.err_func = MSE(beta)
        self.alpha = alpha
        self.batch_size = batch_size
        self.modules = None
        
    def train(self, independent_variables, dependent_variables, epochs):
        modules = [self.l1, self.a1, self.l2, self.a2, self.l3, self.a3, self.err_func] #'''self.a1, self.l2, self.a2,self.l3'''
        self.modules = modules
        if ((type(epochs) == str) and ("inf" in epochs)): epochs = 10**4
        # reshape the dependent variables, so we have Kx1, so later we would compare that vector with output of the size Kx1
        dependent_variables = dependent_variables.reshape(-1, 1)
        
        batch_size = self.batch_size
        total_samples = len(independent_variables)
        
        global_error_arr = []
        for _ in range(epochs):
            batch = 0
            avg_error_epoch = 0
            while True:
                
                #make sure that we don't have a batch with 0 size
                if(batch * batch_size >= total_samples): break 
                
                batch_data = independent_variables[batch * batch_size : (batch + 1) * batch_size]
                depend_batch = dependent_variables[batch * batch_size : (batch + 1) * batch_size]
                
                out_in = batch_data
                for module in modules:
                    if(module == self.err_func):
                        out_in = module.forward(out_in, depend_batch)
                    else:
                        out_in = module.forward(out_in)
                    
                in_out = self.err_func.pl_values
                for module_rev in reversed(modules):
                    if(module_rev == self.err_func):
                        in_out = module_rev.backward(depend_batch, in_out)
                    else:
                        in_out = module_rev.backward(in_out)
                        
                avg_error_epoch += self.err_func.g_error
                batch += 1
            
            global_error_arr.append(avg_error_epoch/batch)
                
            print("Iteration %d" % (_))
            
        global_errorv = global_error_arr[-1]
        print(f"{self.err_func.pl_values} - predicted values")
        print(f"{global_errorv:.2f} - global error")
        return global_error_arr, global_errorv
        

    def test(self, incoming_vector):
        out = incoming_vector
        computat_mod = self.modules.copy()
        computat_mod.pop() #get rid of the last module which is an error function
        
        for module in computat_mod:
            out = module.forward(out)
                
        
        return out
d_appr = True
flag = False

independ_XOR = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
depend_XOR = np.array([1, 1, 0, 0])

independ_OR = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
depend_OR = np.array([1, 1, 0, 1])

independ_AND = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
depend_AND = np.array([0, 0, 0, 1])

train_runs = [(independ_XOR, depend_XOR), (independ_OR, depend_OR), (independ_AND, depend_AND)]
file_1 = open("train_results.txt", "w")
file_2 = open("momentum_impact.txt", "a")
file_3 = open("learning_rate.txt", "a")
plt.grid()
figure_1 = plt.figure(figsize = (8, 5))

if(d_appr):
    for i in range(len(train_runs)):
        features, target = train_runs[i]
        perceptron = MLP(0.021, 0.9, 4)
        global_error_arr, global_errorv = perceptron.train(features, target, 1000)
        name = None
        match i + 1:
            case 1:
                name = "XOR"
            case 2:
                name = "OR"
            case 3:
                name = "AND"
        
        plt.plot([i for i in range(len(global_error_arr))], global_error_arr)
        plt.title(name)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.show()
        file_1.write(f"Name:{name}\nGlobal Error:{global_errorv}\n\n")
elif(not d_appr and flag):
    perceptron_NM = MLP(0.021, 0.0, 3)
    global_error_arr, global_errorv = perceptron_NM.train(independ_XOR, depend_XOR, 1000)
    plt.plot([i for i in range(len(global_error_arr))], global_error_arr)
    plt.title("XOR(Absent momentum)")
    plt.grid(visible = True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()
    file_2.write(f"Momentum(False):{global_errorv}\n\n")
    
    perceptron_PM = MLP(0.021, 0.9, 3)
    global_error_arr, global_errorv = perceptron_PM.train(independ_XOR, depend_XOR, 1000)
    plt.plot([i for i in range(len(global_error_arr))], global_error_arr)
    plt.title("XOR(Present momentum)")
    plt.grid(visible = True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()
    file_2.write(f"Momentum(True):{global_errorv}\n\n")
else:
    perceptron_1 = MLP(0.021, 0.0, 3)
    global_error_arr, global_errorv = perceptron_1.train(independ_XOR, depend_XOR, 1000)
    plt.plot([i for i in range(len(global_error_arr))], global_error_arr)
    plt.title("XOR(Learning rate: 0.021)")
    plt.grid(visible = True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()
    file_3.write(f"XOR(Learning rate: 0.021):{global_errorv}\n\n")
    
    perceptron_2 = MLP(10.0, 0.0, 3)
    global_error_arr, global_errorv = perceptron_2.train(independ_XOR, depend_XOR, 1000)
    plt.plot([i for i in range(len(global_error_arr))], global_error_arr)
    plt.title("XOR(Learning rate: 10.0)")
    plt.grid(visible = True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()
    file_3.write(f"XOR(Learning rate: 10.0):{global_errorv}\n\n")
    


# test_sample = [[1, 0]]
# t_out = perceptron.test(test_sample)
# print(f"{t_out} - predicted value")

file_1.close()
file_2.close()
file_3.close()