import numpy as np
import os
import subprocess
from subprocess import Popen, PIPE

from sklearn.datasets import make_regression

test_list = [(10, 2), (1000,2), (5000,2),
              (10000,2), (5000,10), (1000,100)]

for (samples,features) in test_list:
    print((samples,features))
    (X, y, coefs) = make_regression(n_samples=samples, n_features=features, n_informative=samples, coef=True)
    np.savetxt("X.csv", fmt='%f', delimiter=',', X=X)
    np.savetxt("y.csv", fmt='%f', delimiter=',', X= y.T)
    np.savetxt("theta.csv", fmt='%f', delimiter=',', X= coefs.T)

    cmd = ["./LinearRegressionProject", "X.csv", "y.csv", "theta.csv", "0.01", "50000"]
    
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()
    print("Testing ")
    print(out.decode("utf-8") )
