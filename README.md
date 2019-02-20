# circumplanetary_disks

This repository is used to simulate the light curve produced by circumplanetary disks in any geometry (inclination, tilt, impact parameter) with azimuthally symmetric rings (radii and opacities), with any size star (diameter). This is done using the get_light_curve() function using the following parameters. The length of the eclipse in days (arbitrary value > 0 used to scale rings), location of the centre of the disc w.r.t. the centre of the eclipse (dx, dy) [which will give you the inclination and the tilt of the disk and you have also specified the impact parameter], an image of a star with a given resolution [obtained using limb_darkened_star()], the size of the star in days, and an array containing the radii of the rings and an array containing those rings transmissions. 

It also includes code to take existing light curves and, by defining a few parameters, explore the phase space of disks above providing possible orientations, sizes and optical thicknesses of disks that could produce such a light curve.


