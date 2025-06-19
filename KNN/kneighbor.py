import polars as pl
import numpy as np
import scaler

class KNN:
    def __init__(self, X_train, y_train):
        self.X = np.transpose(X_train).astype(float)
        self.y = y_train
    
    def __nn_indexes(self, X_target, k):
        distances = self.__nn_distances(X_target=X_target)
        nearest_distances = np.full(shape=k, fill_value=np.max(distances))
        nearest_indexes = np.full(shape=k, fill_value=0)
        
        biggest_idx = np.argmax(nearest_distances)
        for index, distance in enumerate(distances):
            if nearest_distances[biggest_idx] > distance:
                nearest_distances[biggest_idx] = distance
                nearest_indexes[biggest_idx] = index
                biggest_idx = np.argmax(nearest_distances)
        return nearest_indexes
    
    def __nn_distances(self, X_target):
        return [self.__nn_distance(np.abs(row - X_target)) for row in self.X]

    def __nn_distance(self, vectors):
        return np.sqrt( np.sum( np.power(vectors, 2)))
    
    # Public
        
    def nn_target_variables(self, X_target, k):
        return self.y[self.__nn_indexes(X_target, k)]
    
    def nn_instances(self, X_target, k):
        return self.X[self.__nn_indexes(X_target, k)]
    
    def nn_distances(self, X_target):
        return self.__nn_distances(X_target)
            

        
    
    



data = np.array([
    [4,4,4,2,2,2],
    [70,120,300,75,200,400],
    [10,4,7,18,8,2]])
s = scaler.Scaler(data)

data = s.scale()
y = np.array([27000,45000,75000,18000,65000,90000])
test_data = np.array([[2.0],[350.0],[9.5]])
test_data = np.transpose(s.scale_new(test_data))
knn = KNN(X_train=data, y_train=y)

print(data)
print(test_data)
print(knn.nn_target_variables(X_target=test_data, k=2))
print(knn.nn_distances(X_target=test_data))
