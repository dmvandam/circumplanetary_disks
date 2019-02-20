#####################################################################
################# CIRCUMPLANETARY DISK SIMULATION ###################
#####################################################################

'''
This module can be used to simulate the light curves produced by a 
star of size d_star [days], and a circumplanetary disk with an
inclination, tilt, impact parameter and an azimuthally symmetric
structure of rings with different radii and opacities
'''

# importing main modules

import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from scipy.signal import convolve
from fit_disk import find_ellipse_parameters

# to compare with Matt's Code
#from exorings import ellipse_strip

#####################################################################
############################ FUNCTIONS ##############################
#####################################################################

def limb_darkened_star(size, u=0.0):
    '''
    This function makes a limb darkened star for convolution with
    the rings

    Parameters
    ----------
    size : float
        the size of the star in pixels (should be an odd number)
    u : float
        limb darkening parameter

    Returns
    -------
    star : array [2-D]
        a size*size array with the intensity percentage of the 
        limb darkened star model
    '''
    # create grid and centre
    star_grid = np.mgrid[:size, :size]
    centre = (size - 1) / 2
    star_grid[0] -= centre
    star_grid[1] -= centre
    # create radius and scale
    radius = np.sqrt(star_grid[0]**2 + star_grid[1]**2)
    radius /= centre
    # create mask
    mask = np.zeros_like(radius)
    mask[(radius > 1.0000001)] = 1.
    radius[(radius > 1.0000001)] = 1.
    # limb darkening
    star = 1. - u * (1 - np.sqrt(1 - radius**2))
    star[(mask > 0)] = 0.
    # normalisation
    star /= np.sum(star)
    return star

def ellipse_grid(sky_grid, te, dx, dy, r, tau):
    '''
    This creates a grid of the rings to be used for a convolution
    with a finite sized star.
    
    Parameters
    ----------
    sky_grid : np.mgrid
        the output of np.mgrid with a grid[0] size of the star in 
        time pixels and a grid[1] size of the eclipse plus 2.5 
        times the star size in time pixels
    te : float
        the duration of the eclipse.
    dx : float
        this is the offset in the x direction of the centre of the
        rings
    dy : float
        this is the offset in the y direction of the centre of the
        rings
    r : array [1-D]
        this contains all the radii of the rings (0 is not necessary)
    tau : array [1-D]
        array must be the same length as r and is the corresponding
        opacity of the ring (i.e. 1 - transmission)
    
    Returns
    -------
    rings : array [2-D]
        an array containing an image of the opacity profile of the
        ring system
    '''
    # transforming grid
    grid = np.copy(sky_grid)
    s = -dx[0]/dy[0]
    #grid[0] -= dy
    grid[1] = sky_grid[1] + s * (sky_grid[0] + dy[0])
    grid[0] = sky_grid[0] + dy[0]
    # shear transformation changes the radius so grid must be scaled
    a,_,_,_ = find_ellipse_parameters(te, dx, dy)
    R = np.sqrt(dy[0]**2 + (te/2.)**2)
    scale = a[0,0]/R
    grid *= scale
    # radius
    rr = np.sqrt(grid[0]**2 + grid[1]**2)
    # making ring model
    ri = 0.0
    rings = np.zeros_like(rr)+0.0
    for i in range(len(r)):
        ro = r[i]
        ring = (rr>=ri) * (rr<ro)
        rings[ring] = tau[i]
        ri = ro
    return rings

def ellipse_gradients(time, dx, dy):
    '''
    This function takes a ring system that is centred on (dx,dy)
    and finds the gradient curve for that ellipse

    Parameters
    ----------
    time : array [1-D]
        an array off all the times the gradient should be calculated
        for.
    dx : float
        this is the offset in the x direction of the centre of the
        rings.
    dy : float
        this is the offset in the y direction of the centre of the
        rings.

    Returns
    -------
    gradients : array [1-D]
        the gradients of the ellipse with the geometry specified in
        the inputs.

    Notes
    -----
    the gradients are defined as the absolute value of the sine of 
    the tangent ("gradient") angle.
    '''
    s = - dx[0] / dy[0]
    num = - s * dy[0] - time
    den = (s**2 + 1) * dy[0] + s * time
    gradients = np.abs(np.sin(np.arctan2(num,den)))
    return gradients

def get_light_curve(te, dx, dy, star, d_star, r, tau):
    '''
    This function takes a ring system with a given geometry
    and finds the light curve after convolving it with a
    star

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
    star : array_like (2-D)
        output of limb_darkened_star, a convolution kernel that is
        used to calculate real light curves.
    d_star : float
        size of the star in days.
    r : array_like (1-D)
        array containing the radii of the rings to be simulated.
        these should be in increasing order and should not contain
        0.
    tau : array_like (1-D)
        array containing the opacity of the rings specified above.
        values should be between 0 and 1 and the array should have
        the same length as r.

    Returns
    -------
    tau_rings : array_like (2-D)
        contains an image of the rings.
    tau_rings_convolved : array_like (2-D)
        is 1 - light curve of the star convolved with the rings.
    plot_data : array_like (2-D)
        array consisting of 4 vectors. [0] = time, [1] = light curve
        if the star was a laser beam, [2] = the light curve with a
        finite sized star, [3] = the gradients measured.

    Notes
    -----
    the gradients are defined as the absolute value of the sine of 
    the tangent ("gradient") angle.
    '''
    # determine grid parameters
    star_y, star_x = star.shape
    dt = d_star / float(star_x - 1) # days/pixel
    nx = np.ceil(2 * (np.amax(r) + 1.25 * d_star) / dt)
    ny = star_y # pixels
    # create grid, shift and scale
    sky_grid = np.mgrid[:ny, :nx]
    yc = (ny - 1) / 2.
    xc = (nx - 1) / 2.
    sky_grid[0] -= yc
    sky_grid[1] -= xc
    sky_grid *= dt
    # creating ring models
    tau_rings = ellipse_grid(sky_grid, te, dx, dy, r, tau)
    tau_rings_convolved = convolve(tau_rings, star, mode='valid')
    # getting relevant parameters
    c = int(yc)
    time = sky_grid[1,c,c:-c]
    lc_td = 1 - tau_rings[c,c:-c]
    lc_tdc = 1 - tau_rings_convolved[0]
    gradients = ellipse_gradients(time, dx, dy)
    # packing data
    plot_data = np.vstack((time,lc_td,lc_tdc,gradients))
    return tau_rings, tau_rings_convolved, plot_data

def vector_rings(r, tau, inc, tilt, dy):
    '''
    This function makes a list of vector patches for plotting the
    ring system with the function: plot_disk_simulation.

    Parameters
    ----------
    r : array_like (1-D)
        array containing the radii of the rings to be simulated.
        these should be in increasing order and should not contain
        0.
    tau : array_like (1-D)
        array containing the opacity of the rings specified above.
        values should be between 0 and 1 and the array should have
        the same length as r.
    inc : float
        inclination of the ring system (degrees).
    tilt : float
        tilt of the ring system w.r.t. the x-axis (degrees)
    dy : float
        y-coordinate of the centre of the ring system w.r.t. the
        centre of the eclipse
    
    Returns
    -------
    rings : list
        list of Ellipse patch objects that can be used by the 
        function plot_disk_simulation to draw the eclipsing system.
    '''
    n = len(r)
    radii = np.flip(r,0)
    radii = np.append(radii,0)
    alpha = np.flip(tau,0)
    alpha = np.append(alpha,1)
    rings = []
    for k in range(n):
        ao = 2*radii[k]
        bo = 2*radii[k]*np.cos(np.deg2rad(inc))
        ai = 2*radii[k+1]
        bi = 2*radii[k+1]*np.cos(np.deg2rad(inc))
        T = alpha[k]
        ellipse = patches.Ellipse((0,-dy),ao,bo,tilt,color='r',alpha=T)
        ellipse_w = patches.Ellipse((0,-dy),ai,bi,tilt,color='w')
        rings.append(ellipse)
        rings.append(ellipse_w)
    return rings


#####################################################################
############################# PLOTTING ##############################
#####################################################################

def plot_disk_simulation(te, dx, dy, r, tau, d_star, plot_data, animate_plot=False, save=False):
    '''
    This function creates a plot that shows the eclipsing system
    The relevant light curve, and the corresponding gradient of
    the light curve.

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
    r : array_like (1-D)
        array containing the radii of the rings to be simulated.
        these should be in increasing order and should not contain
        0.
    tau : array_like (1-D)
        array containing the opacity of the rings specified above.
        values should be between 0 and 1 and the array should have
        the same length as r.
    d_star : float
        size of the star in days.
    plot_data : array_like (2-D)
        array consisting of 4 vectors. [0] = time, [1] = light curve
        if the star was a laser beam, [2] = the light curve with a
        finite sized star, [3] = the gradients measured.
    animate_plot : boolean
        determines whether or not to make an animation of the transit

    Returns
    -------
    Plot or Animation
    
    Notes
    -----
    This function only works for a single (dx, dy) point
    '''
    # determining the angular geometry of the system
    a, b, t, i = find_ellipse_parameters(te, dx, dy)
    # creating the patch objects
    rings = vector_rings(r,tau,i,t,dy)
    # getting the limits of the data
    time = plot_data[0]
    # drawing the star path
    xy = (time[0],-d_star/2.)
    width = time[-1]-time[0]
    star_path = patches.Rectangle(xy,width,d_star,color='g',alpha=0.5)
    ##### creating the figure #####
    # title with all the information pertaining to the disk system
    title1 = 'Eclipse of a Circumplanetary Disk \n'
    title2 = 'Star - $R_{*}$ = %.2f days, b = %.2f days \n'%(d_star/2.,dy[0])
    title3 = 'Disk - $i$ = %.2f$^o$, $\\phi$ = %.2f$^o$, '%(i,t)
    title4 = '$R_{disk}$ = '+str(r)+' days, '
    title5 = '$\\tau$ = '+str(tau)
    title = title1 + title2 + title3 + title4 + title5
    #creating the figure
    fig = plt.figure(figsize=(8,10))
    fig.suptitle(title)
    ax0 = plt.subplot2grid((4,1),(0,0),rowspan=2)
    # plot 1 - transit
    ax0.set_xlim(time[0],time[-1])
    ax0.set_ylim(-d_star,d_star) 
    ax0.set_aspect('equal')
    ax0.set_adjustable('box')
    ax0.set_ylabel('y [days]')
    ax0.axhline(y=0,color='k')
    for ring in rings:
        ax0.add_patch(ring)
    ax0.add_patch(star_path)
    # plot 2 - light curve
    ax1 = plt.subplot2grid((4,1),(2,0),sharex=ax0)
    ax1.set_ylim(-0.05,1.05)
    ax1.set_xlim(time[0],time[-1])
    ax1.set_ylabel('normalised flux')
    # plot 3 - gradients
    ax2 = plt.subplot2grid((4,1),(3,0),sharex=ax0)
    ax2.set_ylim(-0.05,1.05)
    ax2.set_xlim(time[0],time[-1])
    ax2.set_ylabel('gradient')
    ax2.set_xlabel('time [days]')
    def animate(i):
        '''
        This function produces the animation for the plot
        '''
        # clearing the first plot of lines        
        if i == 0:
            for artist in ax1.lines + ax1.collections:
                artist.remove()
            for artist in ax2.lines + ax2.collections:
                artist.remove()
        # clearing the previous star
        try:
            stars[i-1].remove()
            stars[i-1].remove()
        except:
            None
        # moving the star
        star = stars[i]
        ax0.add_artist(star)
        # transmission of rings
        ax1.plot(data[0,:i],data[1,:i],'r-')
        # light curve
        ax1.plot(data[0,:i],data[2,:i],'g-')
        # gradient curve
        ax2.plot(data[0,:i],data[3,:i],'g-')

    if animate_plot == True:
        stars = [patches.Circle((x,0),d_star/2.,color='g',alpha=0.7) for x in plot_data[0]]
        ani = animation.FuncAnimation(fig, animate, frames=len(time),interval=50,repeat_delay=250)
        writer = animation.ImageMagickFileWriter(fps=20)
        if save == True:
            ani.save('eclipse.gif',writer=writer)
        else:
            fig.show()
    else: 
        # add star
        star = patches.Circle((time[0]+d_star,0), d_star/2.,color='g',alpha=0.7)
        ax0.add_patch(star)
        # add ring transmission
        ax1.plot(time,plot_data[1],'r-o')
        ax1.plot(time,plot_data[2],'g-o')
        # add gradient
        ax2.plot(time,plot_data[3],'g-o')
        if save == True:
            fig.savefig('test.png')
        else:
            fig.show()
    return None

#####################################################################
############################### MAIN ################################
#####################################################################

if __name__=="__main__":
    import matplotlib as mpl
    mpl.rc('image',interpolation='nearest',origin='lower left')
    ########### LIMB DARKENED STAR ###########
    print('testing limb darkened star models')
    size = np.array([21,101])
    u = np.array([0,5])
    fig1,axs1 = plt.subplots(2,2)
    fig1.suptitle('Limb Darkened Star for U = 0, U = 5')
    for i in range(len(size)):
        for j in range(len(u)):
            star = limb_darkened_star(size[i],u[j])
            axs1[i,j].imshow(star)
            axs1[i,j].set_xlabel('x [px]')
            axs1[i,j].set_ylabel('y [px]')
    print('')

    ########## ELLIPSE GRID ##########
    print('testing ellipse grid')
    # this if the output of ellipse_grid, which is the
    # first output of get_light_curve
    dx = np.array([0.3,0.6])
    dy = np.array([0.3,0.4])
    te = 0.5
    r = np.array([0.5,1])
    tau = np.array([0.5,1])
    star = limb_darkened_star(21,0)
    d_star = 0.5
    td,_,_ = get_light_curve(te, dx, dy, star, d_star, r, tau)
    _,_,tilt,inclination = find_ellipse_parameters(te, dx, dy)
    ext = (-(np.amax(r)+1.25*d_star),(np.amax(r)+1.25*d_star),-d_star/2., d_star/2.)
    fig2,axs2 = plt.subplots(2,2,figsize=(15,10))
    fig2.suptitle('Disk Images with r = 0.5 and 1, tau = 0.5 and 1')
    for i in range(len(dx)):
        for j in range(len(dy)):
            td,_,_ = get_light_curve(te,dx[i,None],dy[j,None],star,d_star,r,tau)
            params = 'b = %.1f, inc = %.1f, tilt = %.1f'%(dy[j],inclination[j,i],tilt[j,i])
            axs2[i,j].set_xlabel('x [days]')
            axs2[i,j].set_ylabel('y [days]')
            axs2[i,j].set_title(params)
            axs2[i,j].imshow(td,extent=ext)
            axs2[i,j].axhline(y=0,color='r')
            axs2[i,j].axhline(y=-dy[j],color='g')
            axs2[i,j].axvline(x=0,color='r')
    print('')

    ########## ECLIPSE SIMULATION ##########
    print('testing the light curve data')
    # define parameters
    dx = np.array([0.5])
    dy = np.array([0.3])
    r = np.round(np.linspace(0.2,1.2,3),2)
    tau = np.round(np.random.uniform(0,1,3),2)
    td, tdc, data = get_light_curve(te, dx, dy, star, d_star, r, tau)
    # check animation
    print('making animation... (animation is saved)')
    plot_disk_simulation(te, dx, dy, r, tau, d_star, data, True, True)
    # check plot
    print('producing plots')
    plot_disk_simulation(te, dx, dy, r, tau, d_star, data, False) 
    # compare with matt's code
    #a,b,t,i = find_ellipse_parameters(te, dx, dy)
    #print('tilt = %.3f, inclination = %.3f'%(t,i))
    #tr, trc, data_old = ellipse_strip(r, tau, dy[0], 0, i, t, star, d_star)
    #plot_disk_simulation(te, dx, dy, r, tau, d_star, data_old, False)
    plt.show()
    print('')
    print('DONE')


