from numpy import mean,std

def center_data(data):
    '''
    Centers columns by subtracting them by their mean.

    Returns centered data and offset(mean)
    '''
    data_offset = mean(data, axis = 0)
    
    return (data - data_offset), data_offset
