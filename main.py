import csv
import numpy_customed as npc
import matplotlib.pyplot as plt
def main():
    training_dataset = []
    testing_dataset = []
    def fileReading():
        with open("data.csv", "r") as file:
            reader = csv.DictReader(file)
            all_data = list(reader)
            training_dataset.extend(all_data[:300])
            testing_dataset.extend(all_data[300:])
    def split_X_variables_and_Y(dataset):
        X = []
        Y = []
        for data in dataset:
            X_cluster = [1.0] 
            for key, value in data.items():
               if key == 'Y house price of unit area':
                   Y.append(float(value))
               elif key == "No":
                   continue
               else:
                   X_cluster.append(float(value))
            X.append(X_cluster)
        return npc.VectorMatrixOperations(X), npc.VectorMatrixOperations(Y)
    def loss_function(coefficients, X_training_set, Y_training_set):
        return npc.mean((npc.dotProduct(X_training_set, coefficients)-Y_training_set)**2) / 2
    def gradient_descent(coeffcients, x_training_set : npc.VectorMatrixOperations, y_training_set : npc.VectorMatrixOperations):
        return npc.mean(((x_training_set).Transpose() * (npc.dotProduct(x_training_set, coeffcients) - y_training_set)), axis=1)
    def multilinear_regression(coefficients, x_training_set, y_training_set, learning_rate, b1=0.9, b2=0.999, epsilon=1e-8):
        cost_list = []
        prev_cost = 0.0
        m_coef = npc.VectorMatrixOperations([0.0] * (len(coefficients)))
        v_coef = npc.VectorMatrixOperations([0.0] * (len(coefficients)))
        moment_m_coef = npc.VectorMatrixOperations([0.0] * (len(coefficients)))
        moment_v_coef  = npc.VectorMatrixOperations([0.0] * (len(coefficients)))
        new_coefficents = npc.VectorMatrixOperations(coefficients)
        t = 0        
        while True:
            cost = loss_function(new_coefficents, x_training_set, y_training_set)
            cost_list.append(cost)
            if abs(prev_cost - cost) < epsilon:
                break
            prev_cost = cost
            gradients = gradient_descent(new_coefficents, x_training_set, y_training_set)
            t+=1
            m_coef = m_coef * b1 + gradients * (1 - b1)
            v_coef = v_coef * b2 + (1 - b2)* gradients**2
            moment_m_coef = m_coef / (1-b1 ** t)
            moment_v_coef = v_coef / (1-b2 ** t)
            delta = ((learning_rate / (moment_v_coef**0.5) + 1e-8) * (b1 * moment_m_coef + (1-b1) * gradients / (1-b1**t))) 
            new_coefficents = new_coefficents - delta
        return new_coefficents, cost_list
    # Đoạn code này sử dụng phương thức tối ưu Adam
    # Adam optimization algorithm
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    def predict(coefficients, x_testing_set):
        return npc.dotProduct(x_testing_set, coefficients)        
    def start_training():
       fileReading()
       X_trainning_set, Y_trainning_set = split_X_variables_and_Y(training_dataset)
       X_testing_set, Y_testing_set = split_X_variables_and_Y(testing_dataset)
       new_coefficients = [1.0] + [0.0] * (len(X_trainning_set[0]) - 1)
       new_coefficients, cost_list = multilinear_regression(new_coefficients, X_trainning_set, Y_trainning_set, learning_rate=1e-1)
       Y_predict = predict(new_coefficients, X_testing_set)
       fig, axs = plt.subplots(2, figsize=(7,7))
       axs[0].plot(cost_list)
       axs[0].set_xlabel('Số lần lặp')
       axs[0].set_ylabel('Cost')
       axs[0].grid(True)
       axs[1].plot(Y_predict, 'r--', label='Dự đoán')
       axs[1].plot(Y_testing_set, label='Thực tế')
       axs[1].set_ylabel('Giá nhà')
       axs[1].legend(loc='upper left')
       axs[1].grid(True)
       plt.subplots_adjust(hspace = 0.5)
       plt.tight_layout()
       plt.show()
    start_training()
if __name__ == "__main__":
    main()

        
