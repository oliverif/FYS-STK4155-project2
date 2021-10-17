import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import max,min

def plot_surf(x,y,z,x_dp,y_dp):
    '''
    Plots the surface defined by the coordinates x,y,z.
    Assumes x and y is or has been a meshgrid.
    If not meshgrid, reshape back to original shape. 
    '''
    #Ensure flattened array
    if(x.shape!=(y_dp,x_dp)):
        x = x.reshape(y_dp,x_dp)
    if(y.shape!=(y_dp,x_dp)):
        y = y.reshape(y_dp,x_dp)       
    if(z.shape!=(y_dp,x_dp)):
         z = z.reshape(y_dp,x_dp)  


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

def plot_surf_from_X(X,z,x_dp,y_dp):
    '''
    Plotting surface directly from degisn matrix.
    Assumes the feature x is along column 1 of X, and feature y is along column 2.
    '''
    x = X[:,1]
    y = X[:,2]
    return plot_surf(x,y,z, x_dp,y_dp)


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

