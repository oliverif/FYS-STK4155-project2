from imageio import imread
from numpy import meshgrid, linspace,ravel, ones,column_stack



def load_data(fileName, x_dp=100, y_dp=100):
    '''
    Load 'fileName' and
    extract section (0,0) to (x_dp,y_dp) from fileName
    '''
    return imread(fileName)[:y_dp,:x_dp]

def create_normalized_meshgrid(x_dp=100,y_dp=100):
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

def create_poly_design_matrix(x,y=None, degree=2):
    '''
    Creates a polynomial features design matrix with a certain degree.
    With degree 2 the polynomial features are [1, x, y, x^2, xy, y^2]

    Input: Input can be either x = column_stack(x,y) where x and y are assumed flattened,
    or x and y either flattened or as meshgrid. If y is not None, the latter is assumed.
    '''
    if(y is None):
        y = x[:,1]
        x = x[:,0]
    else:
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

   
    
def create_dataset(fileName,N=100,degree=None):
    '''
    Creates design matrix or x,y(depeding on degree) and target data
    Inputs: filename(string), dimension, degree(optional)
    Outputs: Outputs z, and either design matrix X or x and y stacked in columns.
    '''
    z = load_data(fileName,N,N).reshape(-1,1)
    x,y = create_normalized_meshgrid(N,N)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    if(degree):
        X = create_poly_design_matrix(x,y,degree)
        return X,z
    else:
        return column_stack((x.reshape(-1,1),y.reshape(-1,1))),z
