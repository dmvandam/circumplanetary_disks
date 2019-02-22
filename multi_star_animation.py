#####################################################################
####################### MULTI STAR ANIMATION ########################
#####################################################################
'''
This module can be used to make an animation of the eclipse of a ring
system and seeing the effect of changing the size of the star
'''

# importing main modules

from routines_disk_simulation import *
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec, patches
import time


#####################################################################
########################### FUNCTIONS ###############################
#####################################################################

def frame_selection(time, frame_values):
    '''
    finds the arguments of the time array closest to the frame time values
    
    Parameters
    ----------
    time : array (1-D)
        contains the timeseries of the simulation
    frame_values : array (1-D)
        contains the time values corresponding to each frame
    
    Returns
    -------
    frame_indices : array (1-D)
        contains the indices for the relevant frames of the animation
    '''
    frame_indices = []
    for frame_value in frame_values:
        index = np.argmin(np.abs(time - frame_value))
        frame_indices.append(index)
    return frame_indices

def make_multi_star_animation(te, dx, dy, r, tau, star, d_star, savename='eclipse_multi_star.gif'):
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
    d_star : array_like (1-D - len(d_star) <= 3)
        size of the star in days in ascending order.
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
    print('setting up figure')
    # disk parameters
    a, b, tilt, inc = find_ellipse_parameters(te, dx, dy)
    ##### creating the figure #####
    # title with all the information pertaining to the disk system
    title1 = 'Eclipse of a Circumplanetary Disk \n'
    #title2 = 'Star - $R_{*}$ = %.2f days, b = %.2f days \n'%(d_star/2.,dy[0]) # needed later
    title2 = 'Disk - $i$ = %.2f$^o$, $\\phi$ = %.2f$^o$ \n'%(inc,tilt)
    title3 = '$R_{disk}$ = '+str(np.round(r,2))+' days \n '
    title4 = '$\\tau$ = '+str(np.round(tau,2))
    title = title1 + title2 + title3 + title4
    #creating the figure
    fig = plt.figure(figsize=(15,8))
    fig.suptitle(title)
    n = min(len(d_star),3)
    # create axes
    axs = np.zeros((3,0))
    if n >= 1:
        ax0a = plt.subplot2grid((3,n),(0,0))#,rowspan=2)
        ax1a = plt.subplot2grid((3,n),(1,0),sharex=ax0a)
        ax2a = plt.subplot2grid((3,n),(2,0),sharex=ax0a)
        axsa = np.array([ax0a, ax1a, ax2a])
        axs = np.concatenate((axs,axsa[:,None]),1)
    if n >= 2:
        ax0b = plt.subplot2grid((3,n),(0,1))#,rowspan=2)
        ax1b = plt.subplot2grid((3,n),(1,1),sharex=ax0b)
        ax2b = plt.subplot2grid((3,n),(2,1),sharex=ax0b)
        axsb = np.array([ax0b, ax1b, ax2b])
        axs  = np.concatenate((axs,axsb[:,None]),1)
    if n >= 3:
        ax0c = plt.subplot2grid((3,n),(0,2))#,rowspan=2)
        ax1c = plt.subplot2grid((3,n),(1,2),sharex=ax0c)
        ax2c = plt.subplot2grid((3,n),(2,2),sharex=ax0c)
        axsc = np.array([ax0c, ax1c, ax2c])
        axs  = np.concatenate((axs,axsc[:,None]),1)
    # open lists for the animation
    plot_data = []
    all_stars = []
    lims = [[-np.amax(d_star),np.amax(d_star)],[-0.05,1.05],[-0.05,1.05]]
    ylabels = ['y [days]','normalised flux [-]','gradient [-]']
    # set up for animation
    for j in range(n):
        # create ring patches
        rings = vector_rings(r, tau, inc, tilt, dy)
        # getting light curve data
        _, _, data = get_light_curve(te,dx,dy,star,d_star[j],r,tau)
        # introducing a skip factor to reduce equalize the number of frames
        if j == 0:
            frame_times = np.linspace(data[0,0],data[0,-1],101)

        frame_indices = frame_selection(data[0],frame_times)
        ani_data = data[:,frame_indices]
        # appending to plot_data
        plot_data.append(ani_data)
        time = ani_data[0]
        stars = [patches.Circle((x,0),d_star[j]/2.,color='g',alpha=0.7) for x in time]
        all_stars.append(stars)
        for i in range(3):
            # setting limits
            axs[i,j].set_xlim(time[0],time[-1])
            axs[i,j].set_ylim(lims[i])
            # setting labels
            if j == 0:
                axs[i,j].set_ylabel(ylabels[i])
            if i == 2:
                axs[i,j].set_xlabel('time [days]')
            # adding vector graphics
            if i == 0:
                axs[i,j].set_aspect('equal')
                axs[i,j].set_adjustable('box')
                for ring in rings:
                    axs[i,j].add_patch(ring)
                xy_rect = (time[0],-d_star[j]/2.)
                w_rect  = time[-1] - time[0]
                star_path = patches.Rectangle(xy_rect, w_rect, d_star, color='g', alpha=0.5)
                axs[i,j].axhline(y=0, color='k', ls='--')
                axs[i,j].set_title('$D_*$ = %.2f days'%d_star[j])

    def animate(f):
        '''
        This function produces the animation for the plot.
        '''
        print('frame %i/101 \r'%f),
        # clearing the plots of lines
        for j in range(n):
            if f == 0:
                for artist in axs[1,j].lines + axs[1,j].collections:
                    artist.remove()
                for artist in axs[2,j].lines + axs[2,j].collections:
                    artist.remove()
            try:
                all_stars[j][f-1].remove()
                all_stars[j][f-1].remove()
            except:
                None
            # moving the star
            Star = all_stars[j][f]
            axs[0,j].add_artist(Star)
            # transmission of rings
            axs[1,j].plot(plot_data[j][0,:f],plot_data[j][1,:f],'r-')
            # light curve
            axs[1,j].plot(plot_data[j][0,:f],plot_data[j][2,:f],'g-')
            # gradient curve
            axs[2,j].plot(plot_data[j][0,:f],plot_data[j][3,:f],'g-')
    print('creating animation')
    ani = animation.FuncAnimation(fig, animate, interval=5, repeat_delay=250)
    writer = animation.ImageMagickFileWriter(fps=20)
    print('saving animation')
    ani.save(savename,writer=writer)
    return None


#####################################################################
############################### MAIN ################################
#####################################################################

if __name__=="__main__":
    start = time.time()
    print('loading parameters')
    dx = np.array([0.4])
    dy = np.array([0.2])
    te = 0.5
    r = np.linspace(1.15,4.35,5)
    tau = np.random.uniform(0,1,5)
    d_star = np.array([1.0,0.1,0.01])
    star = limb_darkened_star(21,0)
    make_multi_star_animation(te,dx,dy,r,tau,star,d_star)
    end = time.time()
    print('')
    print('done in %.2f seconds'%(end-start))
