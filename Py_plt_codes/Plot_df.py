import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys

nx = 512 
ny = nx
nph = 1 
step = int(input('Enter the step value : '))
#step = 6000000
d8_format = 0 
numsteps = step

filedir0 = "./"
if os.path.isdir(f'{filedir0}/Pngs'):
    pass
else:
    os.mkdir(f'{filedir0}/Pngs')

filedir1 = "./Pngs"

while step>-1 and step<(numsteps+1):
    
    print(step)
    if (d8_format == 1):
        data = np.loadtxt(f"{filedir0}/prof_gp.{step:08d}")
    else:
        data = np.loadtxt(f"{filedir0}/prof_gp.{step}")
    comp = (data[:,2]).reshape(nx,ny)
    if (nph == 4):
        phi = (data[:,3]*1+data[:,4]*2+data[:,5]*3+data[:,6]*4).reshape(nx,ny)
    elif (nph == 1):
        #phi = (data[:,3]).reshape(nx,ny)
        dfdc = (data[:,4]).reshape(nx,ny)
        dfchdphi = (data[:,5]).reshape(nx,ny)
        dgdphi = (data[:,6]).reshape(nx,ny)
        dfeldphi = (data[:,7]).reshape(nx,ny)
        dfdphi = dfchdphi + dgdphi + dfeldphi
 
    plt.close()
    plt.imshow(dfdc,cmap='jet')
    plt.colorbar()
    plt.title(f'dfdc_{step}')
    plt.savefig(f'{filedir1}/dfdc_{step}.png')

    plt.close()
    plt.imshow(dfeldphi,cmap='jet')
    plt.colorbar()
    plt.title(f'dfeldphi_{step}')
    plt.savefig(f'{filedir1}/dfeldphi_{step}.png')
    
    
    print("DONE")
    sys.exit()

    if step == numsteps:
        break;
    if step<numsteps_prof1:
        step += tprof1
    elif step>(numsteps_prof1-1) and step<(numsteps_prof2):
        step += tprof2
    elif step>(numsteps_prof1-2) and step<(numsteps):
        step += tprof3
    #elif step<(numsteps): 
        #step += tprof2
    #plt.show()

## Renaming the plots.
files1 = sorted(os.listdir(filedir1), key=lambda x: int(x.split('_')[1].split('.')[0]))
files2 = sorted(os.listdir(filedir2), key=lambda x: int(x.split('_')[1].split('.')[0]))

print("renaminng the phi_plots")
for i, file in enumerate(files1):
    os.rename(os.path.join(filedir1, file), os.path.join(filedir1, f'phi_{i}.png'))

print("renaminng the comp_plots")
for i, file in enumerate(files2):
    os.rename(os.path.join(filedir2, file), os.path.join(filedir2, f'comp_{i}.png'))

## Making video out of the phi plots.
print("making the phi_plots video")
os.chdir(f"{filedir1}")
command1 = "ffmpeg -framerate 10 -i phi_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p phi.mp4"
subprocess.run(command1, shell=True)

# Making video out of the comp plots.
print("making the comp_plots video")
os.chdir("../comp_plots")
command2 = "ffmpeg -framerate 10 -i comp_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p comp.mp4"
subprocess.run(command2, shell=True)


print("DONE")
