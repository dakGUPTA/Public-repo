import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys

nx = int(input("Enter nx value : "))
ny = nx
nph = 4 
step = int(input('Enter the step value : '))
d8_format = 0 

filedir0 = "./"
if os.path.isdir(f'{filedir0}/Pngs'):
    pass
else:
    os.mkdir(f'{filedir0}/Pngs')

filedir1 = "./Pngs"
  
print(step)
if (d8_format == 1):
    data = np.loadtxt(f"{filedir0}/prof_gp.{step:08d}")
else:
    data = np.loadtxt(f"{filedir0}/prof_gp.{step}")
comp = (data[:,2]).reshape(nx,ny)
if (nph == 4):
    phi =  (data[:,3] + data[:,4] + data[:,5] + data[:,6]).reshape(nx,ny)
    mvphi =  (data[:,3]*1 + data[:,4]*2 + data[:,5]*3 + data[:,6]*4).reshape(nx,ny)
    phi0 = (data[:,3]).reshape(nx,ny)
    phi1 = (data[:,4]).reshape(nx,ny)
    phi2 = (data[:,5]).reshape(nx,ny)
    phi3 = (data[:,6]).reshape(nx,ny)
    phi4 = 1.0 - (phi0 + phi1 + phi2 + phi3)
elif (nph == 1):
    phi = (data[:,3]).reshape(nx,ny)

h_phisum = (phi**3) * (6.0*(phi**2) - 15.0*phi + 10.0)
hphi0 = (phi0**3) * (6.0*(phi0**2) - 15.0*phi0 + 10.0)
hphi1 = (phi1**3) * (6.0*(phi1**2) - 15.0*phi1 + 10.0)
hphi2 = (phi2**3) * (6.0*(phi2**2) - 15.0*phi2 + 10.0)
hphi3 = (phi3**3) * (6.0*(phi3**2) - 15.0*phi3 + 10.0)
#h_phim = (phi4**3) * (6.0*(phi4**2) - 15.0*phi4 + 10.0)
h_phim = 1.0 - h_phisum

hphisum = hphi0 + hphi1 + hphi2 + hphi3
hphim = 1.0 - hphisum

hphim_check = (hphim - h_phim)
hphisum_check = (hphisum - h_phisum)

plt.close()
plt.imshow(hphim_check,cmap='jet')
plt.colorbar()
plt.title('hphim_check')
plt.savefig(f"./Pngs/hphim_check_{step}.png")

plt.close()
plt.imshow(hphisum_check,cmap='jet')
plt.colorbar()
plt.title('hphisum_check')
plt.savefig(f"./Pngs/hphisum_check_{step}.png")
plt.close()

plt.imshow(mvphi,cmap='jet')
plt.colorbar()
plt.title('mvphi')
plt.savefig(f"./Pngs/mvphi_{step}.png")

plt.close()
plt.imshow(h_phim,cmap='jet')
plt.colorbar()
plt.title('h_phim')
plt.savefig(f"./Pngs/h_phim_{step}.png")

plt.close()
plt.imshow(hphim,cmap='jet')
plt.colorbar()
plt.title('hphim')
plt.savefig(f"./Pngs/hphim_{step}.png")

plt.close()
plt.imshow(hphisum,cmap='jet')
plt.colorbar()
plt.title('hphisum')
plt.savefig(f"./Pngs/hphisum_{step}.png")

plt.close()
plt.imshow(h_phisum,cmap='jet')
plt.colorbar()
plt.title('h_phisum')
plt.savefig(f"./Pngs/h_phisum_{step}.png")


