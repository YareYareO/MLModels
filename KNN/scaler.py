import numpy as np

# Setzt voraus, dass im numpy nd array jede SPALTE ein Array ist
class Scaler:
    def __init__(self, X_train):
        self.X_train = X_train.astype(np.float32)
        self.max_values = [np.max(column) for column in X_train]
        self.min_values = [np.min(column) for column in X_train]
        

    def scale(self):
        X_scaled = np.zeros(shape=self.X_train.shape, dtype=np.float32)
        for (i, column) in enumerate(self.X_train):
            
            minimum = self.min_values[i]
            maximum = self.max_values[i]
            match minimum:
                case _ if minimum == maximum:
                    X_scaled[i] = np.zeros(shape=X_scaled[i].shape, dtype=np.float32)
                    self.min_values[i] = 0
                    self.max_values[i] = 0
                    continue
                case _ if minimum == 0:
                    X_scaled[i] = column
                case _ if minimum != 0:
                    X_scaled[i] = column - minimum
                    maximum = maximum - minimum

            assert np.min(X_scaled[i]) == 0, "Minimum ist nicht null, wtf."

            X_scaled[i] = X_scaled[i] / maximum

            assert np.max(X_scaled[i]) == 1, "Maximum ist nicht null, wtf."      
        return X_scaled
        
    def scale_new(self, X_test):
        test_scaled = np.zeros(shape=X_test.shape, dtype=np.float32)
        for (i, column) in enumerate(X_test):
            minimum = self.min_values[i]
            maximum = self.max_values[i]
            if minimum == maximum:
                continue
            test_scaled[i] = (column - minimum) / (maximum - minimum)
        return test_scaled
        
""" data = np.array([[1,2,3,4,5,6,7,8],
                 [-3,-2,-1,0,1,2,3,4],
                 [0,5,6,7,8,9,10,11],
                 [0,0,0,0,0,0,0,0],
                 [7,7,7,7,7,7,7,7],
                 [-10,-9,-15,0,20,40,10,1],
                 [-3,-2,-1,0,1,2,3,4]])
s = Scaler(data)

print(s.scale()) """