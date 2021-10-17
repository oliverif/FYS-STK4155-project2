from imageio import imread
from numpy import meshgrid, linspace,ravel, ones



def load_data(fileName, x_dp, y_dp):
    '''
    Load 'fileName' and
    extract section (0,0) to (x_dp,y_dp) from fileName
    '''
    return imread(fileName)[:y_dp,:x_dp]

def create_normalized_meshgrid(x_dp,y_dp):
    '''
    Creates a meshgrid with x and y between 0 and 1.
    If input coordinates are unequal(non-square meshgrid),
    meshgrid is normalized to the axis with longest range.
    
    Example:
    x_dp = 10, y_dp = 10 -> x and y is interval [0,1]

    x_dp = 10, y_dp = 5 -> x is interval [0,1]
                           y is interval [0,0.5]

    x_dp = 5, y_dp = 10 -> x is interval [0,0.5]
                           y is interval [0,1]
    '''
    norm = max(x_dp,y_dp)
    return meshgrid(linspace(0,x_dp/norm, x_dp),linspace(0,y_dp/norm, y_dp))

def create_poly_design_matrix(x,y, degree):
    '''
    Creates a polynomial features design matrix with a certain degree.
    With degree 2 the polynomial features are [1, x, y, x^2, xy, y^2]
    '''
    #Ensure arrays are flattened
    if len(x.shape) > 1:
        x = ravel(x)
    if len(y.shape) > 1:
        y = ravel(y)

    #Total number of datapoints   
    N = len(x)
    l = int((degree+1)*(degree+2)/2)   # Number of elements in beta
    X = ones((N,l))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

