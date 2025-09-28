import numpy as np

class DeltaRuleBase:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def bipolar_conversion(self, pattern):
        # Convert vector to bipolar (-1,1)
        return np.array(pattern) - 1

    def predict(self, pattern):
        # Returns continuous output (ADALINE)
        return np.dot(self.weights, pattern) + self.bias

class Perceptron(DeltaRuleBase):
    def train(self, patterns, targets, epochs=100):
        for _ in range(epochs):
            for x, t in zip(patterns, targets):
                x_bipolar = self.bipolar_conversion(x)
                output = 1 if self.predict(x_bipolar) >= 0 else -1
                update = self.learning_rate * (t - output)
                self.weights += update * x_bipolar
                self.bias += update
        print(f"Perceptron Final Weights: {self.weights}")
        print(f"Perceptron Final Bias: {self.bias}")

    def classify(self, pattern):
        x_bipolar = self.bipolar_conversion(pattern)
        output = 1 if self.predict(x_bipolar) >= 0 else -1
        return output

class Adaline(DeltaRuleBase):
    def train(self, patterns, targets, epochs=100):
        for _ in range(epochs):
            for x, t in zip(patterns, targets):
                x_bipolar = self.bipolar_conversion(x)
                output = self.predict(x_bipolar)
                error = t - output
                self.weights += self.learning_rate * error * x_bipolar
                self.bias += self.learning_rate * error
        print(f"Adaline Final Weights: {self.weights}")
        print(f"Adaline Final Bias: {self.bias}")

    def classify(self, pattern):
        x_bipolar = self.bipolar_conversion(pattern)
        output = self.predict(x_bipolar)
        return 1 if output >= 0 else -1

if __name__ == "__main__":
    # Input patterns (7x9 pixels flattened to 63-length vectors)
    input_patterns = [
        # Font 1
        [0, 0, 2, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 2, 2, 0, 2, 2, 2], # A
        
        [2, 2, 2, 2, 2, 2, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 2, 2, 2, 2, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         2, 2, 2, 2, 2, 2, 0], # B
        
        [0, 0, 2, 2, 2, 2, 2,
         0, 2, 0, 0, 0, 0, 2,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 0, 2, 2, 2, 2, 0], # C
        
        [2, 2, 2, 2, 2, 0, 0,
         0, 2, 0, 0, 0, 2, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 2, 0,
         2, 2, 2 ,2, 2, 0, 0], # D
        
        [2, 2, 2, 2, 2, 2, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 2, 2, 0, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 0, 0, 0, 0, 0,
         0, 2, 0, 0, 0, 0, 2,
         2, 2, 2, 2, 2, 2, 2], # E
        
        [0, 0, 0, 2, 2, 2, 2,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 2, 0, 0, 0, 2, 0,
         0, 2, 0, 0, 0, 2, 0,
         0, 0, 2, 2, 2, 0, 0], # J
        
        [2, 2, 2, 0, 0, 2, 2,
         0, 2, 0, 0, 2, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 2, 0, 0, 0, 0,
         0, 2, 2, 0, 0, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 0, 0, 2, 0, 0, 
         0, 2, 0, 0, 0, 2, 0,
         2, 2, 2, 0, 0, 2, 2], # K
        
        # Font 2
        [0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0], # A
        
        [2, 2, 2, 2, 2, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 0], # B
        
        [0, 0, 2, 2, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # C
        
        [2, 2, 2, 2, 2, 0, 0,   
         2, 0, 0, 0, 0, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 2, 0,   
         2, 2, 2 ,2, 2, 0, 0], # D
        
        [2, 2, 2, 2, 2, 2, 2,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 2, 2, 2, 2, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 2, 2, 2, 2, 2, 2], # E
        
        [0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # J
        
        [2, 0, 0, 0, 0, 2, 0,   
         2, 0, 0, 0, 2, 0, 0,   
         2, 0, 0, 2, 0, 0, 0,   
         2, 0, 2, 0, 0, 0, 0,   
         2, 2, 0, 0, 0, 0, 0,   
         2, 0, 2, 0, 0, 0, 0,   
         2, 0, 0, 2, 0, 0, 0,   
         2, 0, 0, 0, 2, 0, 0,   
         2, 0, 0, 0, 0, 2, 0], # K
        
        # Font 3
        [0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 2, 2, 2, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 2, 0, 0, 0, 2 ,2], # A
        
        [2, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 0], # B
        
        [0, 0, 2, 2, 2, 0, 2,   
         0, 2, 0, 0, 0, 2, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # C
        
        [2, 2, 2, 2, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 2, 2 ,2, 2, 0, 0], # D
        
        [2, 2, 2, 2, 2, 2, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 2, 0, 0,   
         0, 2, 2, 2, 2, 0, 0,   
         0, 2, 0, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 0, 0,   
         0, 2, 0, 0, 0, 0, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 2], # E
        
        [0, 0, 0, 0, 2, 2, 2,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # J
        
        [2, 2, 2, 0, 0, 2, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 2, 0, 0,   
         0, 2, 0, 2, 0, 0, 0,   
         0, 2, 2, 0, 0, 0, 0,  
         0, 2, 0, 2, 0, 0, 0,   
         0 ,2, 0, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 2, 2, 0, 0, 2, 2], # K
    ]

    # For "B vs not B" classification, targets are 1 for B, -1 otherwise)
    targets = [
        -1, 1, -1, -1, -1, -1, -1,  # Font 1: Only "B" is 1
        -1, 1, -1, -1, -1, -1, -1,  # Font 2
        -1, 1, -1, -1, -1, -1, -1,  # Font 3
    ]

    perceptron = Perceptron(input_size=63, learning_rate=0.01)
    perceptron.train(input_patterns, targets)
    adaline = Adaline(input_size=63, learning_rate=0.01)
    adaline.train(input_patterns, targets)


    # Test with a new pattern (a slightly altered "B")
    some_pattern =  [
        0, 0, 0, 2, 0, 0, 0,   
        0, 0, 0, 2, 0, 0, 0,   
        0, 0, 2, 0, 2, 0, 0,   
        0, 0, 2, 0, 2, 0, 0,   
        0, 2, 0, 0, 0, 2, 0,   
        0, 2, 2, 2, 2, 2, 0,   
        2, 0, 0, 0, 0, 0, 2,   
        2, 0, 0, 0, 0, 0, 2,   
        2, 2, 0, 0, 0, 2 ,2
    ],

    # Make predictions:
    result_p = perceptron.classify(some_pattern)
    result_a = adaline.classify(some_pattern)

    print(f"Perceptron Prediction: {'B' if result_p == 1 else 'Not B'}")
    print(f"Adaline Prediction: {'B' if result_a == 1 else 'Not B'}")

    
    # # Test pattern (Font 3 "A")
    # some_pattern =  [
    #     0, 0, 0, 2, 0, 0, 0,   
    #     0, 0, 0, 2, 0, 0, 0,   
    #     0, 0, 2, 0, 2, 0, 0,   
    #     0, 0, 2, 0, 2, 0, 0,   
    #     0, 2, 0, 0, 0, 2, 0,   
    #     0, 2, 2, 2, 2, 2, 0,   
    #     2, 0, 0, 0, 0, 0, 2,   
    #     2, 0, 0, 0, 0, 0, 2,   
    #     2, 2, 0, 0, 0, 2 ,2
    # ],

    # # Make predictions:
    # result_p = perceptron.classify(some_pattern)
    # result_a = adaline.classify(some_pattern)

    # print(f"Perceptron Prediction: {'B' if result_p == 1 else 'Not B'}")
    # print(f"Adaline Prediction: {'B' if result_a == 1 else 'Not B'}")
