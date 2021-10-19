from numpy import mean

def center_data(data):
    '''
    Centers columns by subtracting them by their mean.

    Returns centered data and offset(mean)
    '''
    offset_data = mean(data, axis = 0)
    return data - offset_data, offset_data