# This script fits disks to data
# It will take a file, calculate gradients and the te
# The output is a single plot with all the possible configs of the
# disk



#####################################################################
########################### MAIN MODULES ############################
#####################################################################

from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import time
import sys
import os


#####################################################################
####################### IMPORTANT FUNCTIONS #########################
#####################################################################

def shear_circle_point(R, s, theta):
    '''
    Transforms a point on a circle to a point on a sheared circle.
    
    The input point is in parametric form, x = R*cos(theta) and
    y = R*sin(theta), and is on a circle. The output point is
    the same point on a sheared circle with the transformation,
    x' = x - s*y, y' = y. Returns the cartesian coordinates on the
    sheared circle.

    Parameters
    ----------
    R : array_like (1-D)
        Array containing radii of circles to be investigated.
    s : array_like (2-D)
        Array contains all the shear factors to be investigated.
    theta : float [rad]
        the angle of the point in the circle that will be transformed
        to the sheared circle.

    Returns
    -------
    xp : array_like (2-D)
        the x-coordinate of the input point in the sheared circle.
    yp : array_like (2-D)
        the y-coordinate of the input point in the sheared circle.
    
    Notes
    -----
    For the purpose of this script it should be noted that:
    
    R is a function of dy and te (R = np.sqrt(dy**2 + (te/2)**2))
    s is a function of dx and dy (s = -dx/dy)
    
    dx and dy are the shifts of the centre of the ellipse w.r.t. the
    centre of the eclipse.

    te is the duration of the eclipse.
    '''
    y = R[:,None]*np.sin(theta)
    x = R[:,None]*np.cos(theta)
    yp = y
    xp = x - s*y
    return xp, yp

def theta_max_min(s):
    '''
    Determines the parametric angle of the location of either the
    semi-major axis or the semi-minor axis of an ellipse sheared by
    the equations: x' = x - s*y, y' = y.

    This is based on the fact that at the vertices and co-vertices
    of an ellipse dr/dtheta = 0.

    Parameters
    ----------
    s : array_like (2-D)
        Array contains all the shear factors to be investigated.

    Returns
    -------
    theta_max_min : array_like (2-D) [rad]
        Array containing the angle of either the semi-major or semi-
        minor axis of an ellipse with corresponding shear factor s.

    Notes
    -----
    For the purpose of this script it should be noted that:
    
    s is a function of dx and dy (s = -dx/dy).

    dx and dy are the shifts of the centre of the ellipse w.r.t. the
    centre of the eclipse.

    This function returns either the location of a co-vertex or a
    vertex (location of the semi-minor or semi-major axis, resp.).
    The two are separated by pi/2 radians.
    '''
    theta_max_min = 0.5 * np.arctan2(2, s)
    return theta_max_min

def find_ellipse_parameters(te,dx,dy):
    '''
    Finds the semi-major axis, a, semi-minor axis, b, the tilt and 
    the inclination of the smallest ellipse that is centred at 
    (dx,dy) w.r.t. the centre of the eclipse with duration te.

    Parameters
    ----------
    te : float
        duration of the eclipse.
    dx : array_like (1-D)
        Array containing all the circle centres shifted by dx in 
        the x direction.
    dy : array_like (1-D)
        Array containing all the circle centres shifted by dy in
        the y direction.

    Returns
    -------
    a : array_like (2-D)
        Array containing all the semi-major axes of the ellipses
        investigated. i.e. with their centres at (dx,dy).
    b : array_like (2-D)
        Array containing all the semi-minor axes of the ellipses
        investigated. i.e. with their centres at (dx,dy).
    tilt : array_like (2-D)
        Array containing all the tilt angles of the ellipses
        investigated. i.e. with their centres at (dx,dy). Tilt is
        the angle of the semi-major axis w.r.t. the x-axis. [deg]
    inclination : array_like (2-D)
        Array containing all the inclination angles of the ellipses
        investigated. i.e. with their centres at (dx,dy). 
        Inclination is based on the ratio of semi-minor to semi-major
        axis. [deg]

    '''
    # determining the radii and shear factors
    R = np.sqrt(dy**2 + (te/2.)**2)
    s = -dx[None,:]/dy[:,None]
    # find position of (co-) vertices
    theta1 = theta_max_min(s)
    theta2 = theta1 + np.pi/2
    x1, y1 = shear_circle_point(R, s, theta1)
    x2, y2 = shear_circle_point(R, s, theta2)
    # find the semi-major and semi-minor axes
    R1 = np.sqrt(x1**2 + y1**2)
    R2 = np.sqrt(x2**2 + y2**2)
    a = np.maximum(R1,R2)
    b = np.minimum(R1,R2)
    # determine the inclination
    inclination = np.arccos(b/a)
    # determine the tilt
    tilt = np.arctan2(y1,x1) # assuming R1 > R2
    tilt_mask = R2 > R1 # find where above is not true
    tilt = tilt + tilt_mask*np.pi/2 # at ^ locations add np.pi/2
    return a, b, np.rad2deg(tilt), np.rad2deg(inclination)

def find_ellipse_slopes(te, dx, dy):
    '''
    Finds the slopes of the tangents to the smallest ellipse centred
    at (dx,dy) at the locations of the eclipse.
    
    i.e. finds the tangents of the ellipse at (-te/2,-dy) and 
    (te/2, dy).

    This is converted to an angle so that the range is [0,1].

    Parameters
    ----------
    te : float
        duration of the eclipse.
    dx : array_like (1-D)
        Array containing all the circle centres shifted by dx in 
        the x direction.
    dy : array_like (1-D)
        Array containing all the circle centres shifted by dy in
        the y direction.
    
    Returns
    -------
    slope1 : array_like (2-D)
        Array containing all the slopes of the tangents of the left
        hand side. i.e. at (-te/2,-dy). Note that the slopes are
        computed as the absolute value of the sine of the arctangent
        of dy/dx.
    slope2 : array_like (2-D)
        Array containing all the slopes of the tangents of the right
        hand side. i.e. at (te/2,-dy). Note that the slopes are
        computed as the absolute value of the sine of the arctangent
        of dy/dx.

    Notes
    -----
    The reason the slopes are modified is because:
    The arctangent of dy/dx prevents the no solution boundary of a
        of a vertical line.
    The sine of this angle makes clear what the maximum and minimum
        slopes are by transforming the range to [-1,1].
    The absolute value is because we are not interested in negative
        slopes as we are astrophysically unable to determine, whether
        a slope is positive or negative, thus limiting our range to
        [0,1].
    '''
    # determining the shear factors
    s = -dx[None,:]/dy[:,None]
    # determining coordinates on circle
    y = -dy
    x1 = -te/2.
    x2 = te/2.
    # shearing circle
    yp = y
    x1p = x1 - s*y[:,None]
    x2p = x2 - s*y[:,None]
    # find slopes based on:
    # R**2 = xp**2 + yp**2 = (x - s*y)**2 + y**2
    # dy/dx = (-s*yp - xp)/((s**2 + 1)*yp + s*xp)
    # phi = arctan(dy/dx), slope = abs(sin(phi))
    # left side
    dy1 = -s*yp[:,None] - x1p
    dx1 = (s**2 + 1)*yp[:,None] + s*x1p
    slope1 = np.arctan2(dy1,dx1)
    # right side
    dy2 = -s*yp[:,None] - x2p
    dx2 = (s**2 + 1)*yp[:,None] + s*x2p
    slope2 = np.arctan2(dy2,dx2)
    return slope1, slope2

def reflect_quadrants(array):
    '''
    Input array is the top right quadrant (QI). This is reflected
    across the relevant axes to create a four quadrant array.

    Parameters
    ----------
    array : array_like (2-D)
        m x n array containing for example the semi-major axis, the,
        semi-minor axis, the inclination or the tilt of the ellipses.

    Returns
    -------
    new_array : array_like (2-D)
        2m x 2n array containing same information as above but
        flipped and reflected to fill up four quadrants

    Example
    -------
    array = np.array([[0,1],[1,1]])
    new_array = np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]])

             [1,1,1,1]
    [0,1] -> [1,0,0,1]
    [1,1] -> [1,0,0,1]
             [1,1,1,1]

    Notes
    -----
    Above example may seem confusing, but this is because of the way
    python indices work, y is increasing from top to bottom so QI is
    at the bottom right, instead of top right, and is reflected w.r.t.
    the y-axis
    '''
    # create new array
    dy, dx = array.shape
    new_array = np.zeros((2*dy, 2*dx))
    # reflections and copying
    new_array[:dy,:dx] = np.flipud(np.fliplr(array)) # lower left
    new_array[:dy,dx:] = np.flipud(array) # lower right
    new_array[dy:,:dx] = np.fliplr(array) # upper left
    new_array[dy:,dx:] = array # upper right (original quadrant)
    return new_array

def reflect_slopes(slope_left, slope_right):
    '''
    Input array is the top right quadrant (QI) of the ellipse slopes
    on the left and on the right. The ellipse is reflected across the
    relevant axes, which means that the slope_left and slope_right
    arrays are flipped (about an axis), copied and switched (i.e.
    slope left can be found in the new_slope_right and vice versa)
    to create a four quadrant array.

    Parameters
    ----------
    slope_left : array_like (2-D)
        m x n array containing the left hand slopes of the ellipses
        investigated.
    slope_right : array_like (2-D)
        m x n array containing the right hand slopes of the ellipses
        investigated.

    Returns
    -------
    new_slope_left : array_like (2-D)
        2m x 2n array filled up as four quadrants using the necessary
        flipping, copying and switching.
    new_slope_right : array_like (2-D)
        2m x 2n array filled up as four quadrants using the necessary
        flipping, copying and switching.

    Example
    -------
    slope_left = np.array([[0,1],[1,1]])
    slope_right = np.array([[1,1],[1,1]])
    new_slope_left, new_slope_right = 
    np.array([[1,1,-1,-1],[1,1,0,-1],[-1,-1,0,1],[-1,-1,1,1]]),
    np.array([[1,1,-1,-1],[1,0,-1,-1],[-1,0,1,1],[-1,-1,1,1]])

                    [ 1, 1,-1,-1]   [ 1, 1,-1,-1]
    [0,1] [1,1]  -> [ 1, 0, 0,-1]   [ 1, 0,-1,-1]
    [1,1],[1,1]  -> [-1,-1, 0, 1]   [-1, 0, 1, 1]
                    [-1,-1, 1, 1] , [-1,-1, 1, 1]

    Notes
    -----
    Above example may seem confusing, but this is because of the way
    python indices work, y is increasing from top to bottom so QI is
    at the bottom right, instead of top right, and is reflected w.r.t.
    the y-axis.
    
    The other thing to consider is that if you reflect an ellipse
    over the y-axis (i.e. flip the x coordinate) the left slope 
    becomes the right slope and vice versa.
    '''
    # determining the shape
    dy,dx = slope_left.shape
    # creating new arrays
    new_slope_left = np.zeros((2*dy, 2*dx))
    new_slope_right = np.zeros((2*dy, 2*dx))
    # below U is upper, L is lower, l is left,r is right
    # reflections, copying and switching - LEFT
    new_slope_left[dy:,dx:] = slope_left # Ur (original quadrant)
    new_slope_left[dy:,:dx] = -np.fliplr(slope_right) # Ul (switched)
    new_slope_left[:dy,dx:] = -np.flipud(slope_left) # Lr
    new_slope_left[:dy,:dx] = np.flipud(np.fliplr(slope_right)) # Ll
    # reflections, copying and switching - RIGHT
    new_slope_right[dy:,dx:] = slope_right
    new_slope_right[dy:,:dx] = -np.fliplr(slope_left)
    new_slope_right[:dy,dx:] = -np.flipud(slope_right)
    new_slope_right[:dy,:dx] = np.flipud(np.fliplr(slope_left))
    return new_slope_left, new_slope_right

def investigate_ellipses(te, xmax, ymax, nx=50, ny=50, ymin=1e-3):
    '''
    Investigates the full parameter space for an eclipse of duration
    te with centres at [-xmax,xmax] (2*nx), [-ymax,ymax] (2*ny)

    Parameters
    ----------
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    nx : integer
        number of steps to investigate in the x direction. i.e.
        investigated space is np.linspace(-xmax, xmax, 2*nx).
    ny : integer
        number of steps to investigate in the y direction. i.e.
        investigated space is np.linspace(-ymax, ymax, 2*ny).
    ymin : float
        this is necessary because the shear parameter s = -dx/dy
        so dy != 0

    Returns
    -------
    a : array_like (2-D)
        Array containing the semi-major axes of the investigated
        ellipses.
    b : array_like (2-D)
        Array containing the semi-minor axes of the investigated
        ellipses.
    tilt : array_like (2-D)
        Array containing the tilt angles [deg] of the investigated
        ellipses. Tilt is the angle of the semi-major axis w.r.t.
        the x-axis
    inclination : array_like (2-D)
        Array containing the inclination angles [deg] of the
        investigated ellipses. Inclination is the angle obtained
        from the ratio of the semi-minor to semi-major axis
    slope_left : array_like (2-D)
        Array containing all the slopes of the left ellipse edge at
        the eclipse height. Note that this slope is defined as the
        absolute value of the sine of the arctangent of dy/dx.
    slope_right
        Array containing all the slopes of the right ellipse edge at
        the eclipse height. Note that this slope is defined as the
        absolute value of the sine of the arctangent of dy/dx.

    Notes
    -----
    This function investigates the phase space available and the
    returned arrays can be used to make plots, gain insight and
    boundaries can be applied to these grids to limit the number of
    valid solutions for the given eclipse profile.
    '''
    # creating phase space
    dy = np.linspace(ymin, ymax, ny)
    dx = np.linspace(0, xmax, nx)
    # investigating phase space
    a, b, tilt, inclination = find_ellipse_parameters(te, dx, dy)
    slope_left, slope_right = find_ellipse_slopes(te, dx, dy)
    # filling the quadrants
    radii = reflect_quadrants(a)
    tilt = reflect_quadrants(tilt)
    inclination = reflect_quadrants(inclination)
    slope_left, slope_right = reflect_slopes(slope_left, slope_right)
    slope_left = np.abs(np.sin(slope_left))
    slope_right = np.abs(np.sin(slope_right))
    return radii, tilt, inclination, slope_left, slope_right


#####################################################################
######################### BOUND FUNCTIONS ###########################
#####################################################################

def mask(mask_array, lower_limit, upper_limit, arrays):
    '''
    This function applies a lower and upper limit mask to mask_array
    and applies this to all the arrays.

    Parameters
    ----------
    mask_array : array_like (2-D)
        Array containing values on which to apply limits
    lower_limit : float
        lower limit of mask_array that is considered acceptable.
    upper_limit : float
        upper limit of mask_array that is considered acceptable.
    arrays : list of array_like
        arrays (including mask_array) that will be masked

    Returns
    -------
    arrays : list of array_like
        arrays (including mask_array) that have been masked according
        to the lower and upper limits imposed.
    '''
    # creating mask
    mask_lower = mask_array > lower_limit
    mask_upper = mask_array < upper_limit
    mask_total = mask_lower * mask_upper
    # applying mask
    for i in range(len(arrays)):
        # makes masked out = 0
        arrays[i] = arrays[i] * mask_total
        # for some arrays 0 is a valid option so make 0's nans
        arrays[i][mask_total == False] = np.nan
    return arrays

"""
SUBJECT TO CHANGE
def gradient_mask(arrays, te, t_measured, grad_measured, xmax, ymax, nx=50, ny=50, ymin=1e-3, max_roll=0.1):
    '''
    Masks out all the solutions where the measured gradients exceed
    the theoretical gradients.

    Parameters
    ----------
    arrays : list of array_like
        Arrays that will be masked.
    te : float
        Duration of the eclipse.
    t_measured : array_like (1-D)
        Array containing the time of a gradient measured in a light
        curve file.
    grad_measured : array_like (1-D)
        Array containing the gradient measured in a light curve file.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    nx : integer
        number of steps to investigate in the x direction. i.e.
        investigated space is np.linspace(-xmax, xmax, 2*nx).
    ny : integer
        number of steps to investigate in the y direction. i.e.
        investigated space is np.linspace(-ymax, ymax, 2*ny).
    ymin : float
        this is necessary because the shear parameter s = -dx/dy
        so dy != 0
    max_roll : float
        To compare the measured gradients to the simulated gradients,
        the midpoint of the eclipse must be known. Here values for
        the midpoint are shifted point by point until the max_roll has
        been obtained.

    Returns
    -------
    arrays : list of array_like
        Arrays that have been masked according to the theoretical
        gradients at t_measured (grad_measured <= grad)
    '''
    # constructing phase space
    dx = np.linspace(0, xmax, nx)
    dx = np.concatenate((-np.flip(dx,0),dx),0)
    dy = np.linspace(ymin, ymax, ny)
    dy = np.concatenate((-np.flip(dy,0),dy),0)
    # determining the shear factor
    s = -dx[None,:]/dy[:,None]
    # in ring-space the gradients are at (x,y) = (t_measured,-dy)
    x = t_measured
    y = -dy
    # determining the theoretical gradient at all t_measured
    DY = -s[:,:,None]*y[:,None,None] - x[None,None,:]
    DX = (s[:,:,None]**2+1)*y[:,None,None]+s[:,:,None]*x[None,None,:]
    grad = np.abs(np.sin(np.arctan2(DY,DX)))
    # Dealing with the uncertainty of the eclipse midpoint
    max_dt = max_roll * te
    centre = np.argmin(np.abs(t_measured))
    min_roll = np.argmin(np.abs(t_measured+max_dt))-centre
    max_roll = np.argmin(np.abs(t_measured-max_dt))-centre
    roll_arr = np.arange(min_roll,max_roll+1e-3).astype(np.int)
    roll_arr = np.array([0])
    # Checking through all the values
    grad_mask_full = np.zeros(s.shape)
    for r in roll_arr:
        # now this is compared to the measured gradients
        grad_measured_rolled = np.roll(grad_measured,r)
        grad_check = np.sum((grad >= grad_measured_rolled),2)
        # a (dx,dy) position of the centre is only possible if all
        # theoretical gradients are greater than the measured ones
        grad_mask = grad_check == len(t_measured)
        print(np.sum(grad_mask))
        grad_mask_full += grad_mask
    print(np.sum(grad_mask_full))
    grad_mask_full = grad_mask_full > 0

    for i in range(len(arrays)):
        arrays[i] = arrays[i]*grad_mask_full
        arrays[i][grad_mask_full==False] = np.nan
    return arrays
"""

#####################################################################
######################## PLOTTING FUNCTIONS #########################
#####################################################################

def plot_radii(radii, te, xmax, ymax, lvls="none", vmax=15, root='', extra=''):
    '''
    Plots the radii of the investigated ellipses.
    
    Parameters
    ----------
    radii : array_like (2-D)
        Array containing the radii of the investigated ellipses.
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    lvls : list [default = None]
        list contains the contours levels to overlay on the plot
    vmax : float [default = 15]
        maximum value in the colourmap
    root : string [default = '']
        string containing the path where figure is saved
    extra : string [default '']
        string appended to filename to differentiate it from other
        runs of this function

    Returns
    -------
    A figure saved at root+'te_%.3f_radii%s.png'%(te,extra).
    '''
    ny, nx = radii.shape
    fig = plt.figure(figsize=(11,9))
    plt.xlabel('x offset [days]')
    plt.ylabel('y offset [days]')
    plt.xticks(np.linspace(0,nx-1,11),np.linspace(-xmax,xmax,11))
    plt.yticks(np.linspace(0,ny-1,11),np.linspace(-ymax,ymax,11))
    plt.title('Minimum Radii for an Eclipse Time of %.3f days'%te)
    if lvls == "none":
        c = plt.contour(radii,colors='r')
    elif isinstance(lvls,np.int):
        levels = np.linspace(0,vmax,lvls+1)[1:]
        c = plt.contour(radii,levels=levels,colors='r')
    else:
        c = plt.contour(radii,levels=lvls,colors='r')
    plt.gca().clabel(c,c.levels,inline=True,fmt='%.1f',fontsize=8)
    plt.imshow(radii,cmap='viridis',vmin=0,vmax=vmax,origin='lower left')
    plt.colorbar()
    fig.savefig(root+'te_%.3f_radii%s.png'%(te,extra))
    return None

def plot_inclination(inclination, te, xmax, ymax, lvls="none", root='', vmax=90, extra=''):
    '''
    Plots the inclination angles of the investigated ellipses.
    
    Parameters
    ----------
    inclination : array_like (2-D)
        Array containing the inclination angles of the investigated
        ellipses.
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    lvls : list [default = None]
        list contains the contours levels to overlay on the plot
    vmax : float [default = 90]
        maximum value in the colourmap
    root : string [default = '']
        string containing the path where figure is saved
    extra : string [default '']
        string appended to filename to differentiate it from other
        runs of this function

    Returns
    -------
    A figure saved at root+'te_%.3f_inclination%s.png'%(te,extra).
    '''
    ny, nx = inclination.shape
    fig = plt.figure(figsize=(11,9))
    plt.xlabel('x offset [days]')
    plt.ylabel('y offset [days]')
    plt.title('Inclination of Disk with an Eclipse Time of %.3f days'%te)
    plt.xticks(np.linspace(0,nx-1,11),np.linspace(-xmax,xmax,11))
    plt.yticks(np.linspace(0,ny-1,11),np.linspace(-ymax,ymax,11))
    if lvls == "none":
        c = plt.contour(inclination,colors='r')
    elif isinstance(lvls,np.int):
        levels = np.linspace(0,vmax,lvls+1)[1:]
        c = plt.contour(inclination,levels=levels,colors='r')
    else:
        c = plt.contour(inclination,levels=lvls,colors='r')
    plt.gca().clabel(c,c.levels,inline=True,fmt='%.1f',fontsize=8)
    plt.imshow(inclination,cmap='viridis',origin='lower left',vmin=0,vmax=vmax)
    plt.colorbar()
    fig.savefig(root+'te_%.3f_inclination%s.png'%(te,extra))
    return None

def plot_tilt(tilt, te, xmax, ymax, lvls="none", root='', vmax=180, extra=''):
    '''
    Plots the tilt angles of the investigated ellipses.
    
    Parameters
    ----------
    tilt : array_like (2-D)
        Array containing the tilt angles of the investigated ellipses.
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    lvls : list [default = None]
        list contains the contours levels to overlay on the plot
    vmax : float [default = 180]
        maximum value in the colourmap. minimum value is -vmax
    root : string [default = '']
        string containing the path where figure is saved
    extra : string [default '']
        string appended to filename to differentiate it from other
        runs of this function

    Returns
    -------
    A figure saved at root+'te_%.3f_tilt%s.png'%(te,extra).
    '''
    ny, nx = tilt.shape
    fig = plt.figure(figsize=(11,9))
    plt.xlabel('x offset [days]')
    plt.ylabel('y offset [days]')
    plt.title('Tilt of Disk with an Eclipse Time of %.3f days'%te)
    plt.xticks(np.linspace(0,nx-1,11),np.linspace(-xmax,xmax,11))
    plt.yticks(np.linspace(0,ny-1,11),np.linspace(-ymax,ymax,11))
    if lvls == "none":
        c = plt.contour(tilt,colors='r')
    elif isinstance(lvls,np.int):
        levels = np.linspace(-vmax,vmax,lvls+1)[1:]
        c = plt.contour(tilt,levels=levels,colors='r')
    else:
        c = plt.contour(tilt,levels=lvls,colors='r')
    plt.gca().clabel(c,c.levels,inline=True,fmt='%.1f',fontsize=8)
    plt.imshow(tilt,cmap='viridis',origin='lower left',vmax=vmax)
    plt.colorbar()
    fig.savefig(root+'te_%.3f_tilt%s.png'%(te,extra))
    return None

def plot_slope(slope, te, xmax, ymax, loc, lvls="none", vmax=1, root='', extra=''):
    '''
    Plots the slope, with modified range [0,1] of the investigated
    ellipses.
    
    Parameters
    ----------
    slope : array_like (2-D)
        Array containing the modified slopes of the investigated 
        ellipses (left or right).
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    loc : string
        string used in the title and savename of the plot to show
        whether these are slopes of the left or right edge of the
        disk.
    lvls : list [default = None]
        list contains the contours levels to overlay on the plot
    vmax : float [default = 1]
        maximum value in the colourmap.
    root : string [default = '']
        string containing the path where figure is saved
    extra : string [default '']
        string appended to filename to differentiate it from other
        runs of this function

    Returns
    -------
    A figure saved at root+'te_%.3f_slope%s%s.png'%(te, loc, extra).
    '''

    ny,nx = slope.shape
    fig = plt.figure(figsize=(11,9))
    plt.xlabel('x offset [days]')
    plt.ylabel('y offset [days]')
    plt.title('Slope of %s Edge of Disk with an Eclipse Time of %.3f days'%(loc,te))
    plt.xticks(np.linspace(0,nx-1,11),np.linspace(-xmax,xmax,11))
    plt.yticks(np.linspace(0,ny-1,11),np.linspace(-ymax,ymax,11))
    if lvls == "none":
        c = plt.contour(slope,colors='r')
    elif isinstance(lvls,np.int):
        levels = np.linspace(0,vmax,lvls+1)[1:]
        c = plt.contour(slope,levels=levels,colors='r')
    else:
        c = plt.contour(slope,levels=lvls,colors='r')
    plt.gca().clabel(c,c.levels,inline=True,fmt='%.1f',fontsize=8)
    plt.imshow(slope,cmap='viridis',origin='lower left',vmin=0,vmax=vmax)
    plt.colorbar()
    fig.savefig(root+'te_%.3f_slope_%s%s.png'%(te, loc, extra))
    return None

#####################################################################
############################### MAIN ################################
#####################################################################

if __name__=="__main__":
    ##### testing shear_circle_point #####
    print('testing shear_circle_point: circles become ellipses')
    print('and points on circle are sheared to ellipse')
    print('')
    # set params
    theta = np.linspace(0,2*np.pi,101)
    dx = np.linspace(0,1,2)
    dy = np.linspace(0.3,0.5,2)
    R = np.linspace(1,2,2)
    s = -dx[None,:]/dy[:,None]
    # arrays for ellipses and circles
    x = np.zeros(s.shape+theta.shape)
    y = np.zeros(s.shape+theta.shape)
    xc = np.zeros(R.shape+theta.shape)
    yc = np.zeros(R.shape+theta.shape)
    # draw circles
    for i in range(len(R)):
        circle = shear_circle_point(R[None,i],np.array([0]),theta)
        xc[i] = circle[0][0]
        yc[i] = circle[1][0]
    # draw ellipses
    for i, th in enumerate(theta):
        x[:,:,i], y[:,:,i] = shear_circle_point(R,s,th)
    # limits
    xlim = np.amax(np.abs(x))
    ylim = np.amax(np.abs(y))
    lim = np.ceil(max(xlim,ylim))
    # plot figure
    fig1,ax1 = plt.subplots(2,2,sharex=True,sharey=True,figsize=(15,10))
    fig1.suptitle("shear_circle_point")
    points = np.random.randint(0,len(theta),2)
    for i in range(2):
        for j in range(2):
            ax1[i,j].set_xlim(-lim,lim)
            ax1[i,j].set_ylim(-lim,lim)
            ax1[i,j].set_xlabel('x')
            ax1[i,j].set_ylabel('y')
            ax1[i,j].set_aspect('equal')
            ax1[i,j].set_title('R = %.2f, s = %.2f'%(R[i],s[i,j]))
            ax1[i,j].plot(x[i,j],y[i,j],'r-')
            ax1[i,j].plot(x[i,j,points],y[i,j,points],'rx')
            ax1[i,j].plot(xc[i],yc[i],'g-')
            ax1[i,j].plot(xc[i,points],yc[i,points],'gx')
            ax1[i,j].axhline(y=yc[i,points[0]],color='k',linestyle='--')
            ax1[i,j].axhline(y=yc[i,points[1]],color='k',linestyle='--')
        
    ##### testing theta_max_min #####
    print('testing theta_max_min: where the angle of the point')
    print('of the semi-major and semi-minor axis are determined')
    print('')
    # finding the angles
    thetac1 = theta_max_min(np.zeros_like(s))
    thetac2 = thetac1 + np.pi/2
    theta1 = theta_max_min(s)
    theta2 = theta1 + np.pi/2
    # finding the points
    x1,y1 = shear_circle_point(R,s,theta1)
    x2,y2 = shear_circle_point(R,s,theta2)
    xc1,yc1 = shear_circle_point(R,np.zeros_like(s),thetac1)
    xc2,yc2 = shear_circle_point(R,np.zeros_like(s),thetac2)
    # plot figure
    fig2,ax2 = plt.subplots(2,2,sharex=True,sharey=True,figsize=(15,10))
    fig2.suptitle("theta_max_min")
    for i in range(2):
        for j in range(2):
            ax2[i,j].set_xlim(-lim,lim)
            ax2[i,j].set_ylim(-lim,lim)
            ax2[i,j].set_xlabel('x')
            ax2[i,j].set_ylabel('y')
            ax2[i,j].set_aspect('equal')
            ax2[i,j].set_title('$R$ = %.1f, $s$ = %.1f, $\\theta_1$ = %.1f$^o$, $\\theta_2$ = %.1f$^o$'%(R[i],s[i,j],np.rad2deg(theta1[i,j]),np.rad2deg(theta2[i,j])))
            ax2[i,j].plot(x[i,j],y[i,j],'r-')
            ax2[i,j].plot(x1[i,j],y1[i,j],'rx')
            ax2[i,j].plot(x2[i,j],y2[i,j],'rx')
            ax2[i,j].plot(xc[i],yc[i],'g-')
            ax2[i,j].plot(xc1[i],yc1[i],'gx')
            ax2[i,j].plot(xc2[i],yc2[i],'gx')

    ##### find_ellipse_parameters #####
    print('testing find_ellipse_parameters: finding the semi-major,')
    print('semi-minor axis and tilt. this is tested by creating a ')
    print('patch which draws an ellipse using these parameters')
    print('')
    # eclipse duration to get back the radii R determined at the top
    te = np.sqrt(4*(R**2 - dy**2))
    # calculate a, b, tilt, inclination
    a,b,tilt,inclination = find_ellipse_parameters(te, dx, dy)
    # plot figure
    fig3,ax3 = plt.subplots(2,2,sharex=True,sharey=True,figsize=(15,10))
    fig3.suptitle("find_ellipse_parameters")
    for i in range(2):
        for j in range(2):
            ellipse = patches.Ellipse((0,0),2*a[i,j],2*b[i,j],tilt[i,j],color='r',alpha=0.2)
            ax3[i,j].set_xlim(-lim,lim)
            ax3[i,j].set_ylim(-lim,lim)
            ax3[i,j].set_xlabel('x')
            ax3[i,j].set_ylabel('y')
            ax3[i,j].set_aspect('equal')
            ax3[i,j].set_title('$R$ = %.1f, $s$ = %.1f, $a$ = %.1f, $b$ = %.1f, $\\phi$ = %.1f$^o$, $i$ = %.1f$^o$'%(R[i],s[i,j],a[i,j],b[i,j],tilt[i,j],inclination[i,j]))
            ax3[i,j].plot(x[i,j],y[i,j],'r-')
            ax3[i,j].plot(x1[i,j],y1[i,j],'rx')
            ax3[i,j].plot(x2[i,j],y2[i,j],'rx')
            ax3[i,j].plot(xc[i],yc[i],'g-')
            ax3[i,j].plot(xc1[i],yc1[i],'gx')
            ax3[i,j].plot(xc2[i],yc2[i],'gx')
            ax3[i,j].add_patch(ellipse)

    ##### find_ellipse_slopes #####
    print('testing find_ellipse_slopes: slope at the impact parameter')
    print('')
    # plot figure
    fig4,ax4 = plt.subplots(2,2,sharex=True,sharey=True,figsize=(15,10))
    fig4.suptitle("find_ellipse_slopes")
    X = np.linspace(-0.2,0.2,3)
    for i in range(2):
        for j in range(2):
            # te is a float input so calculated here
            ml,mr = find_ellipse_slopes(te[i],dx[None,j],dy[None,i])
            ml = np.tan(ml)
            mr = np.tan(mr)
            ax4[i,j].set_xlim(-te[i],te[i])
            ax4[i,j].set_ylim(-te[i],te[i])
            ax4[i,j].set_aspect('equal')
            ax4[i,j].set_xlabel('x')
            ax4[i,j].set_ylabel('y')
            ax4[i,j].set_title('R = %.1f, s = %.1f, $m_{left}$ = %.2f, $m_{right}$ = %.1f'%(R[i],s[i,j],ml[0],mr[0]))
            ax4[i,j].plot(x[i,j]+dx[j],y[i,j]+dy[i],'r-')
            ax4[i,j].plot(dx[j],dy[i],'ro')
            ax4[i,j].plot([-te[i]/2.,te[i]/2.],[0,0],'k-o')
            ax4[i,j].plot(-te[i]/2.+X,ml[0]*X,'k--')
            ax4[i,j].plot(te[i]/2.+X,mr[0]*X,'k--')

    ##### reflect_quadrants #####
    print('testing reflect_quadrants')
    print('')
    array = np.arange(9).reshape((3,3)).astype(np.float)
    new_array = reflect_quadrants(array)
    fig5,ax5 = plt.subplots(1,2,figsize=(15,10))
    fig5.suptitle('reflect_quadrants')
    for i in range(2):
        ax5[i].set_xlabel('x [pix]')
        ax5[i].set_ylabel('y [pix]')
    p1 = ax5[0].imshow(array,origin='lower left')
    ax5[0].set_title('array')
    p2 = ax5[1].imshow(new_array,origin='lower left')
    ax5[1].set_title('new array')
    plt.colorbar(p1,ax=ax5[0])
    plt.colorbar(p2,ax=ax5[1])
    
    ##### mask #####
    print('testing mask')
    print('')
    masked_array = mask(new_array,0.5,3.5,[new_array])
    fig6, ax6 = plt.subplots(1,2,figsize=(15,10))
    fig6.suptitle('mask')
    for i in range(2):
        ax6[i].set_xlabel('x [pix]')
        ax6[i].set_ylabel('y [pix]')
    p1 = ax6[0].imshow(new_array,origin='lower left')
    ax6[0].set_title('array')
    p2 = ax6[1].imshow(masked_array[0],origin='lower left')
    ax6[1].set_title('masked array (0.5-3.5)')
    plt.colorbar(p1,ax=ax6[0])
    plt.colorbar(p2,ax=ax6[1])

    ##### reflect_slopes #####
    print('testing reflect_slopes')
    print('')
    ml = np.copy(array)
    mr = -np.copy(array)+8
    rml,rmr = reflect_slopes(ml,mr)
    fig7,ax7 = plt.subplots(2,2,figsize=(15,10))
    fig7.suptitle('reflect_slopes')
    for i in range(2):
        for j in range(2):
            ax7[i,j].set_xlabel('x [pix]')
            ax7[i,j].set_ylabel('y [pix]')
    p1 = ax7[0,0].imshow(ml,origin='lower left')
    plt.colorbar(p1,ax=ax7[0,0])
    ax7[0,0].set_title('slope left')
    p2 = ax7[0,1].imshow(mr,origin='lower left')
    plt.colorbar(p2,ax=ax7[0,1])
    ax7[0,1].set_title('slope right')
    p1a = ax7[1,0].imshow(rml,origin='lower left')
    plt.colorbar(p1a,ax=ax7[1,0])
    ax7[1,0].set_title('reflected slope left')
    p2a = ax7[1,1].imshow(rmr,origin='lower left')
    plt.colorbar(p2a,ax=ax7[1,1])
    ax7[1,1].set_title('reflected slope right')
    
    plt.show()

