import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import max,min

def plot_3Dsurface(x,y,z,x_dp,y_dp):
    
    #Ensure flattened array
    if(len(x.shape)<2):
        x = x.reshape(y_dp,x_dp)
    if(len(y.shape)<2):
        y = y.reshape(y_dp,x_dp)       
    if(len(z.shape)<2):
         z = z.reshape(y_dp,x_dp)  



    #If x is given as a design matrix

    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #ax.zaxis.set_ticks_position('top')
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

def plot_colormap(z,x_dp,y_dp,color):
    fig, ax = plt.subplots()
    fig.suptitle('Terrain over Norway 1')
    im = ax.imshow(z.reshape(y_dp,x_dp), cmap=color)

    fig.colorbar(im, shrink=1, aspect=15) 
    im.set_clim(-500,2000)
    #ax.plot(z.reshape(y_dp,x_dp))
    plt.xlabel = 'X'
    plt.ylabel = 'Y'
    return fig

