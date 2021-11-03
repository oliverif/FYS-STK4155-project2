from numpy import sum,mean,size

def R2(z_data, z_model):
    '''
    Returns the R2 score for z_data and z_model
    '''
    return 1 - sum((z_data - z_model) ** 2) / sum((z_data - mean(z_data)) ** 2)

def MSE(z_data,z_model):
    '''
    Returns the MSE for z_data and z_model
    '''
    n = size(z_model)
    return sum((z_data-z_model)**2)/n

def scores(z_data,z_model):
    '''
    Returns the MSE and R2 scole for z_data and z_model
    '''
    return MSE(z_data,z_model),R2(z_data,z_model)


