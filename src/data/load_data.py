from imageio import imread

def load_data(fileName, x_dp, y_dp):
    return imread(fileName)[:y_dp,:x_dp]
