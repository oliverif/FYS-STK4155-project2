import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from utils.utils import *
from math import ceil

def plot_surf(x,y,z):
    '''
    Plots the surface defined by the coordinates x,y,z.
    Assumes x and y is or has been a meshgrid.
    If not meshgrid, reshape back to original shape. Use only
    the x_dp^2 or y_dp^2 first datapoints to ensure array can be reshaped
    into square.
    '''
    x_dp=int(np.sqrt(x.shape[0]))
    y_dp=int(np.sqrt(y.shape[0]))

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
    ax.set_xlim3d(np.min(x), np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlim3d(np.min(z), np.max(z))
    #Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=15,pad = 0.2) 
    plt.close()   
    return ax

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

def plot_plots(plots,params):
    '''
    Adds the curve(s) of the given parameter 'arrays'
    as subfigure to 'fig'.

    Inputs
    --------
    plots: list of (n_plots) dictionaries
        For each dictionary plot every curve it contains
        in one subplot. 
    
    labels: ndarray of str with len(n_plots,n_curves)
        Contains the labels for each curve
    
    fig: matplotlib.figure.Figure object
        The master figure to add this subfigure to.
    
    pos: int
        The index of the subplot, i.e position in the
        figure.
    '''

    n_plots = len(plots)
    n_rows = ceil(n_plots/2)
    n_cols = min(2,len(plots))
    print(n_cols)
    fig, axs = plt.subplots(n_rows,n_cols,figsize=(min(15,n_cols*7.5),5*n_rows))

    if (isinstance(axs,np.ndarray)):

        for i,ax in enumerate(axs.ravel()):
            ax = plot_curves(plots[i][0], params, ax)
            ax.set_xlabel(plots[i][1][0])
            ax.set_ylabel(plots[i][1][1])
            ax.set_xticks(params)
    else:
        axs = plot_curves(plots[0][0],params,axs)
        axs.set_xlabel(plots[0][1][0])
        axs.set_ylabel(plots[0][1][1])
        axs.set_xticks(params)
    plt.close()

    return fig

def plot_curves(curves,params , ax):
    '''
    Plots all curves in curves in one plot
    
    Input
    --------
    curves: dict{label:curve}
        Dictionary containing the curves to be plotted
        keyed on their labels
    
    ax: Axes object
        ax used to plot
    '''

    for (label, vals) in curves.items():
        ax.semilogx(params,vals,label=label)
    ax.legend()
    return ax

def plot_train_test_mse_r2(mse_train, mse_test, r2_train,r2_test,params,param_label):
    MSE_dict = {'MSE train':mse_train, 'MSE test':mse_test}
    MSE_labels = (param_label,'squared error')

    R2_dict = {'R2 train':r2_train, 'R2 test':r2_test}
    R2_labels = (param_label,'R2 score')
    plots = [(MSE_dict,MSE_labels),(R2_dict,R2_labels)]
    return plot_plots(plots,params)
