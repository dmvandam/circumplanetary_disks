from routines_disk_simulation import limb_darkened_star, get_light_curve, vector_rings
from routines_disk_fitting import find_ellipse_parameters
from exorings import ellipse_strip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

'''
def make_grad(time,flux,lvl=1):
     grad = []
     date = []
     for i in range(lvl,len(time)-lvl):
         x = time[i-lvl:i+lvl+1]
         f = flux[i-lvl:i+lvl+1]
         p0, _ = curve_fit(line,x,f)
         grad.append(p0[0])
         date.append(time[i])
     grad = np.array(grad)
     date = np.array(date)
     g = np.abs(np.sin(np.arctan2(grad,1)))
     return date, g

def plot_gradients(data):
     fig,ax = plt.subplots(2,1,sharex=True)
     fig.suptitle('convolution effects')
     ax[0].set_ylabel('relative flux [-]')
     ax[0].plot(data[0],data[2],'r-')
     ax[1].set_xlabel('time w.r.t. t_mid [days]')
     ax[1].set_ylabel('gradient G')
     ax[1].plot(data[0],data[3],'r-',label='theory')
     for lvl in range(1,10):
         d,g = make_grad(data[0],data[2],lvl)
         ax[1].plot(d,g,label='calc - n = %i'%(2*lvl+1))
     ax[1].legend(loc='center left')
     fig.show()
'''

def plot_comparison(te, dx, dy, star_px, d_star, r, tau):
     '''
     This function compares the output of matt kenworthy's exorings functions
     to dirk van dam's and creates a single plot to demonstrate the differences
     and prints the difference in the relevant arrays

     Parameters
     ----------
     te : float
        duration of the eclipse.
     dx : array (1-D)
        the x shift of the centre of the disk w.r.t. the centre of the eclipse.
     dy : array (1-D)
        the y shift of the centre of the disk w.r.t. the centre of the eclipse.
     star_px : int
        the resolution of the convolution grid
     d_star : float
        the size of the star in days
     r : array (1-D)
        contains the radii of the rings (in ascending order) does not include 0
     tau : array (1-D)
        contains the transmission of the rings above (from 0 - 1)

     Returns
     -------
     plot containing the differences between matt kenworthy's exorings routines
     and dirk van dam's routines
     '''
     # create star
     star = limb_darkened_star(star_px)
     # get parameters for exorings (works with angles instead of centre shifts)
     a,b,t,i = find_ellipse_parameters(te, dx, dy)
     # simulate light curve (dirk)
     tr, trc, data = get_light_curve(te, dx, dy, star, d_star, r, tau)
     # simulate light curve (matt)
     td, tdc, data_old = ellipse_strip(r, tau, dy[0], 0, i, t, star, d_star)                                                 
     #### Create Figure #####
     fig,ax = plt.subplots(4,2,sharex=True,figsize=(20,15))    
     ### BOTH ###                
     xy_rect = (data[0,0],-d_star/2.)                          
     w_rect = data[0,-1]-data[0,0]                             
     star_path = patches.Rectangle(xy_rect,w_rect,d_star,color='g',alpha=0.2)                                            
     xy_star = (data[0,0]+d_star/2.,0)                         
     star_disk = patches.Circle(xy_star,d_star/2.,color='g')   
     ### DIRK ###                 
     # imshow of rings grid                                    
     ext = (data[0,0],data[0,-1],-d_star/2.,d_star/2.)      
     ax[0,0].imshow(tr,origin='lower left',extent=ext)      
     ax[0,0].axhline(y=0,color='k')                         
     ax[0,0].axhline(y=dy[0],color='g',linestyle='--')
     ax[0,0].axvline(x=0,color='k')                   
     ax[0,0].axvline(x=dx[0],color='g',linestyle='--')
     ax[0,0].set_ylabel('y size [days]')              
     ax[0,0].set_title('Ellipse Grid - Dirk')         
     ax[0,0].set_xlim(data[0,0],data[0,-1])           
     ax[0,0].set_ylim(-d_star/2.,d_star/2.)           
     # vector rings                                   
     rings = vector_rings(r, tau, i, t, -dy)
     ax[1,0].set_aspect('equal')                 
     for ring in rings:                        
         ax[1,0].add_patch(ring)               
     ax[1,0].add_patch(star_path)              
     ax[1,0].add_patch(star_disk)              
     ax[1,0].set_ylim(-d_star,d_star)          
     ax[1,0].set_ylabel('y size [days]')
     ax[1,0].axhline(y=0,color='k')  
     ax[1,0].axhline(y=dy[0],color='g',linestyle='--')
     ax[1,0].axvline(x=0,color='k')     
     ax[1,0].axvline(x=dx[0],color='g',linestyle='--')
     # light curve (convolved and laser beam)         
     ax[2,0].plot(data[0],data[1],'r-o',alpha=0.5)              
     ax[2,0].plot(data[0],data[2],'g-x',alpha=0.5)
     ax[2,0].set_ylabel('relative flux')
     ax[2,0].set_ylim(-0.05,1.05)
     # gradient curve
     ax[3,0].plot(data[0],data[3],'g-x',alpha=0.5)
     ax[3,0].set_ylabel('gradient')
     ax[3,0].set_ylim(-0.05,1.05)
     ax[3,0].set_xlabel('time [days]')
     ### MATT ###                 
     # imshow of rings grid                                    
     ext = (data_old[0,0],data_old[0,-1],-d_star/2.,d_star/2.)      
     ax[0,1].imshow(td,origin='lower left',extent=ext)      
     ax[0,1].axhline(y=0,color='k')                         
     ax[0,1].axhline(y=dy[0],color='g',linestyle='--')
     ax[0,1].axvline(x=0,color='k')                   
     ax[0,1].axvline(x=dx[0],color='g',linestyle='--')
     ax[0,1].set_ylabel('y size [days]')              
     ax[0,1].set_title('Ellipse Grid - Matt')         
     ax[0,1].set_xlim(data_old[0,0],data_old[0,-1])           
     ax[0,1].set_ylim(-d_star/2.,d_star/2.)           
     # vector rings                                   
     rings_old = vector_rings(r, tau, i, t, -dy)
     star_path_old = patches.Rectangle(xy_rect,w_rect,d_star,color='g',alpha=0.2)                                            
     star_disk_old = patches.Circle(xy_star,d_star/2.,color='g')   
     ax[1,1].set_aspect('equal')                 
     for ring_old in rings_old:                        
         ax[1,1].add_patch(ring_old)               
     ax[1,1].add_patch(star_path_old)
     ax[1,1].add_patch(star_disk_old)
     ax[1,1].set_ylim(-d_star,d_star)          
     ax[1,1].set_ylabel('y size [days]')
     ax[1,1].axhline(y=0,color='k')  
     ax[1,1].axhline(y=dy[0],color='g',linestyle='--')
     ax[1,1].axvline(x=0,color='k')     
     ax[1,1].axvline(x=dx[0],color='g',linestyle='--')
     # light curve (convolved and laser beam)         
     ax[2,1].plot(data_old[0],data_old[1],'r-o',alpha=0.5)              
     ax[2,1].plot(data_old[0],data_old[2],'g-x',alpha=0.5)
     ax[2,1].set_ylabel('relative flux')
     ax[2,1].set_ylim(-0.05,1.05)
     # gradient curve
     ax[3,1].plot(data_old[0],data_old[3],'g-x',alpha=0.5)
     ax[3,1].set_ylabel('gradient')
     ax[3,1].set_ylim(-0.05,1.05)
     ax[3,1].set_xlabel('time [days]')
     
     ### show ###
     plt.show()

     ##### print statements #####

     print('the largest difference in the ellipse grid matrix is: ')
     print(np.amax(np.abs(tr-td)))
     print('')
     print('the largest difference in the light curve data matrix is:')
     print(np.amax(np.abs(data-data_old)))

     return None

if __name__=="__main__":
    te = 0.5
    dx = np.array([0.5])
    dy = np.array([0.3])
    star_px = 21
    d_star = 0.5
    r = np.linspace(2,4,5)
    tau = np.array([1,0,1,0,1])
    plot_comparison(te, dx, dy, star_px, d_star, r, tau)

