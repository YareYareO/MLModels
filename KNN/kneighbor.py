import polars as pl
import numpy as np

class KNN:
    def __init__(self, X_train, y_train):
        self.X = X_train.astype(float)
        self.y = y_train
    
    def nearest_neighbors(self, X_target, k):
        distances = [self.distance(np.abs(row - X_target)) for row in self.X]
        nearest_distances = np.full(shape=k, fill_value=np.max(distances))
        nearest_indexes = np.full(shape=k, fill_value=0)
        
        biggest_idx = np.argmax(nearest_distances)
        for index, distance in enumerate(distances):
            if nearest_distances[biggest_idx] > distance:
                nearest_distances[biggest_idx] = distance
                nearest_indexes[biggest_idx] = index
                biggest_idx = np.argmax(nearest_distances)
        return nearest_indexes
        
        
            
    def distance(self, vectors):
        return np.sqrt( np.sum( np.power(vectors, 2)))
        
    
    



data = np.transpose(np.array([
    [4,4,4,2,2,2],
    [70,120,300,75,200,400],
    [10,4,7,18,8,2]]))
y = np.array([27000,45000,75000,18000,65000,90000])

knn = KNN(X_train=data, y_train=y)
print(knn.nearest_neighbors(X_target=np.array([2.0,350.0,9.5]), k=2))