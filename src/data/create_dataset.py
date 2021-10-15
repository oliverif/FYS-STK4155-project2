from imageio import imread
from numpy import meshgrid, linspace,ravel, ones



def load_data(fileName, x_dp, y_dp):
    return imread(fileName)[:y_dp,:x_dp]

def create_normalized_meshgrid(x_dp,y_dp):
    norm = max(x_dp,y_dp)
    return meshgrid(linspace(0,x_dp/norm, x_dp),linspace(0,y_dp/norm, y_dp))

def create_poly_design_matrix(x,y, degree):
    #Ensure arrays are flattened
    if len(x.shape) > 1:
        x = ravel(x)
    if len(y.shape) > 1:
        y = ravel(y)
        
    N = len(x)
    l = int((degree+1)*(degree+2)/2)   # Number of elements in beta
    X = ones((N,l))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

