import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys

nx = int(input('Enter nx value : '))
ny = nx
step = int(input('Enter the step value : '))
#nph = int(input("Enter Num of Variants:")) 
d8_format = 0 
numsteps = step

filedir0 = "./"
if os.path.isdir(f'{filedir0}/Pngs'):
    pass
else:
    os.mkdir(f'{filedir0}/Pngs')

filedir1 = "./Pngs"

s11 = np.zeros((nx,ny))
s22 = np.zeros((nx,ny))
s12 = np.zeros((nx,ny))

while step>-1 and step<(numsteps+1):
    numsteps = step-1
    
    #print(step)
    #if (d8_format == 1):
    #    data = np.loadtxt(f"{filedir0}/prof_gp.{step:08d}")
    #else:
    #    data = np.loadtxt(f"{filedir0}/prof_gp.{step}")
    #comp = (data[:,2]).reshape(nx,ny)
    #phi = (data[:,3]).reshape(nx,ny)

    exx = np.loadtxt('e11.dat').reshape(nx,ny)
    eyy = np.loadtxt('e22.dat').reshape(nx,ny)
    exy = np.loadtxt('e12.dat').reshape(nx,ny)

    sxx = np.loadtxt('s11.dat').reshape(nx,ny)
    syy = np.loadtxt('s22.dat').reshape(nx,ny)
    sxy = np.loadtxt('s12.dat').reshape(nx,ny)
    
    #s11 = (data[:,4]).reshape(nx,ny)
    #s22 = (data[:,5]).reshape(nx,ny)
    #s33 = (data[:,6]).reshape(nx,ny)
    #s12 = (data[:,7]).reshape(nx,ny)

    #s11 = 0.5*(sxx+syy) + np.sqrt((0.5*(sxx-syy))**2 + sxy**2)
    #s22 = 0.5*(sxx+syy) - np.sqrt((0.5*(sxx-syy))**2 + sxy**2)
    #s12 = (s11 - s22)/2
 
    #plt.close()
    #plt.imshow(phi,origin='lower',cmap='jet')
    #plt.colorbar()
    #plt.title(f'phi_{step}')
    #plt.savefig(f'{filedir1}/phi_{step}.png')

    #plt.close()
    #plt.imshow(comp,origin='lower',cmap='jet')
    #plt.colorbar()
    #plt.title(f'comp_{step}')
    #plt.savefig(f'{filedir1}/comp_{step}.png')
    
    #plt.close()
    #plt.imshow(s11,origin='lower',cmap='jet')
    #plt.colorbar()
    #plt.title(f's11_{step}')
    #plt.savefig(f'{filedir1}/s11_{step}.png')
    #
    #plt.close()
    #plt.plot((s11[nx//2,nx//2:]*160.22),'r')
    #plt.grid()
    #plt.title(f's11_lp_{step}')
    #plt.savefig(f'{filedir1}/s11_lp_{step}.png')
    
    plt.close()
    plt.imshow(sxx*160.22,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'sxx_{step}')
    plt.savefig(f'{filedir1}/sxx_{step}.png')
    
    plt.close()
    plt.plot((sxx[nx//2,:]*160.22),'r')
    plt.grid()
    plt.title(f'sxx_lp_{step}')
    plt.savefig(f'{filedir1}/sxx_lp_{step}.png')

    plt.close()
    plt.imshow(syy*160.22,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'syy_{step}')
    plt.savefig(f'{filedir1}/syy_{step}.png')
    
    plt.close()
    plt.plot((syy[nx//2,:]*160.22),'k')
    plt.grid()
    plt.title(f'syy_lp_{step}')
    plt.savefig(f'{filedir1}/syy_lp_{step}.png')
    
    #plt.close()
    #plt.imshow(s22,origin='lower',cmap='jet')
    #plt.colorbar()
    #plt.title(f's22_{step}')
    #plt.savefig(f'{filedir1}/s22_{step}.png')
    #
    #plt.close()
    #plt.plot((s22[nx//2,nx//2:]*160.22),'r')
    #plt.grid()
    #plt.title(f's22_lp_{step}')
    #plt.savefig(f'{filedir1}/s22_lp_{step}.png')
    
    #plt.close()
    #plt.imshow(szz,origin='lower',cmap='jet')
    #plt.colorbar()
    #plt.title(f'szz_{step}')
    #plt.savefig(f'{filedir1}/szz_{step}.png')
    
    #plt.close()
    #plt.plot((szz[nx//2,nx//2:]*160.22),'r')
    #plt.grid()
    #plt.title(f'szz_lp_{step}')
    #plt.savefig(f'{filedir1}/szz_lp_{step}.png')
    
    plt.close()
    plt.imshow(sxy*160.22,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'sxy_{step}')
    plt.savefig(f'{filedir1}/sxy_{step}.png')
    
    plt.close()
    plt.plot((sxy[:,50]*160.22),'b')
    plt.grid()
    plt.title(f'sxy_lp_{step}')
    plt.savefig(f'{filedir1}/sxy_lp_{step}.png')
     
    #plt.close()
    #plt.imshow(s12,origin='lower',cmap='jet')
    #plt.colorbar()
    #plt.title(f's12_{step}')
    #plt.savefig(f'{filedir1}/s12_{step}.png')
    #
    #plt.close()
    #plt.plot((s12[nx//2,nx//2:]*160.22),'r')
    #plt.grid()
    #plt.title(f's12_lp_{step}')
    #plt.savefig(f'{filedir1}/s12_lp_{step}.png')
    
    plt.close()
    plt.imshow(exx*160.22,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'exx_{step}')
    plt.savefig(f'{filedir1}/exx_{step}.png')
    
    plt.close()
    plt.plot((exx[nx//2,:]*160.22),'r')
    plt.grid()
    plt.title(f'exx_lp_{step}')
    plt.savefig(f'{filedir1}/exx_lp_{step}.png')

    plt.close()
    plt.imshow(eyy*160.22,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'eyy_{step}')
    plt.savefig(f'{filedir1}/eyy_{step}.png')
    
    plt.close()
    plt.plot((eyy[nx//2,:]*160.22),'k')
    plt.grid()
    plt.title(f'eyy_lp_{step}')
    plt.savefig(f'{filedir1}/eyy_lp_{step}.png')
    
    plt.close()
    plt.imshow(exy*160.22,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'exy_{step}')
    plt.savefig(f'{filedir1}/exy_{step}.png')
    
    plt.close()
    plt.plot((exy[:,50]*160.22),'b')
    plt.grid()
    plt.title(f'exy_lp_{step}')
    plt.savefig(f'{filedir1}/exy_lp_{step}.png')
    

print("DONE")
