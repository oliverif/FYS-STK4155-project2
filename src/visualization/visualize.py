import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import max,min, sqrt
from utils.utils import *

def plot_surf(x,y,z):
    '''
    Plots the surface defined by the coordinates x,y,z.
    Assumes x and y is or has been a meshgrid.
    If not meshgrid, reshape back to original shape. Use only
    the x_dp^2 or y_dp^2 first datapoints to ensure array can be reshaped
    into square.
    '''
    x_dp=int(sqrt(x.shape[0]))
    y_dp=int(sqrt(y.shape[0]))

    #Ensure flattened array
    if(x.shape!=(y_dp,x_dp)):
        x = x[:x_dp**2].reshape(y_dp,x_dp)
    if(y.shape!=(y_dp,x_dp)):
        y = y[:y_dp**2].reshape(y_dp,x_dp)       
    if(z.shape!=(y_dp,x_dp)):
         z = z[:y_dp*x_dp].reshape(y_dp,x_dp)  


    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111, projection='3d')


    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.setp(ax.zaxis.get_majorticklabels(), ha='left')

    ax.set_xlabel('X',fontsize=15)
    ax.set_ylabel('Y',fontsize=15)
    ax.set_zlabel('height',labelpad=25,fontsize=15)
    #Set limits to axis according to min and max
    ax.set_xlim3d(min(x), max(x))
    ax.set_ylim3d(min(y), max(y))
    ax.set_zlim3d(min(z), max(z))
    #Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=15,pad = 0.2)    
    return fig

def plot_surf_from_X(X,z):
    '''
    Plotting surface directly from degisn matrix.
    Assumes the feature x is along column 1 of X, and feature y is along column 2.
    '''
    X,z = sort_surface(X,z)
    x = X[:,1]
    y = X[:,2]

    return plot_surf(x,y,z)


def plot_colormap(z,x_dp,y_dp,color):
    '''
    Plots the colormap of the selected terrain area
    '''
    fig, ax = plt.subplots()
    fig.suptitle('Terrain over Norway 1')
    im = ax.imshow(z.reshape(y_dp,x_dp), cmap=color)

    fig.colorbar(im, shrink=1, aspect=15) 
    plt.xlabel = 'X'
    plt.ylabel = 'Y'
    return fig

