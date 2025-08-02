import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys

nx = int(input('Enter nx value : '))
ny = nx
step = int(input('Enter the step value : '))
nph = 4 
d8_format = 0 
numsteps = step
df = 0
engs = 1

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
    
    print(step)
    if (d8_format == 1):
        data = np.loadtxt(f"{filedir0}/prof_gp.{step:08d}")
    else:
        data = np.loadtxt(f"{filedir0}/prof_gp.{step}")
    comp = (data[:,2]).reshape(nx,ny)
    if (nph == 4):
        phi = (data[:,3]*1+data[:,4]*2+data[:,5]*3+data[:,6]*4).reshape(nx,ny)
    elif (nph == 1):
        phi = (data[:,3]).reshape(nx,ny)
   
    if (df == 1):
        dfeldphi1 = (data[:,7]).reshape(nx,ny)
        dfeldphi2 = (data[:,8]).reshape(nx,ny)
        dfeldphi3 = (data[:,9]).reshape(nx,ny)
        dfeldphi4 = (data[:,10]).reshape(nx,ny)
        sxx = (data[:,11]).reshape(nx,ny)
        syy = (data[:,12]).reshape(nx,ny)
        szz = (data[:,13]).reshape(nx,ny)
        sxy = (data[:,14]).reshape(nx,ny)
        exx = (data[:,15]).reshape(nx,ny)
        eyy = (data[:,16]).reshape(nx,ny)
        ezz = (data[:,17]).reshape(nx,ny)
        exy = (data[:,18]).reshape(nx,ny)

        s11 = 0.5*(sxx+syy) + np.sqrt((0.5*(sxx-syy))**2 + sxy**2)
        s22 = 0.5*(sxx+syy) - np.sqrt((0.5*(sxx-syy))**2 + sxy**2)
        s12 = (s11 - s22)/2

    ftotal = (data[:,-3]).reshape(nx,ny)
    felas1 = (data[:,-2]).reshape(nx,ny)
    felas2 = (data[:,-1]).reshape(nx,ny)
        
    plt.close()
    plt.imshow(phi,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'phi_{step}')
    plt.savefig(f'{filedir1}/phi_{step}.png')

    plt.close()
    plt.imshow(comp,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'comp_{step}')
    plt.savefig(f'{filedir1}/comp_{step}.png')
    
    if (df == 1):
        plt.close()
        plt.imshow(dfeldphi1,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'dfeldphi1_{step}')
        plt.savefig(f'{filedir1}/dfeldphi1_{step}.png')
        
        plt.close()
        plt.imshow(dfeldphi2,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'dfeldphi2_{step}')
        plt.savefig(f'{filedir1}/dfeldphi2_{step}.png')
        
        plt.close()
        plt.imshow(dfeldphi3,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'dfeldphi3_{step}')
        plt.savefig(f'{filedir1}/dfeldphi3_{step}.png')
        
        plt.close()
        plt.imshow(dfeldphi4,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'dfeldphi4_{step}')
        plt.savefig(f'{filedir1}/dfeldphi4_{step}.png')
        
        plt.close()
        plt.imshow(s11,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f's11_{step}')
        plt.savefig(f'{filedir1}/s11_{step}.png')
        
        plt.close()
        plt.plot((s11[nx//2,nx//2:]*160.22),'r')
        plt.grid()
        plt.title(f's11_lp_{step}')
        plt.savefig(f'{filedir1}/s11_lp_{step}.png')
        
        plt.close()
        plt.imshow(sxx,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'sxx_{step}')
        plt.savefig(f'{filedir1}/sxx_{step}.png')
        
        plt.close()
        plt.plot((sxx[nx//2,:]*160.22),'r')
        plt.grid()
        plt.title(f'sxx_lp_{step}')
        plt.savefig(f'{filedir1}/sxx_lp_{step}.png')

        plt.close()
        plt.imshow(syy,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'syy_{step}')
        plt.savefig(f'{filedir1}/syy_{step}.png')
        
        plt.close()
        plt.plot((syy[nx//2,:]*160.22),'k')
        plt.grid()
        plt.title(f'syy_lp_{step}')
        plt.savefig(f'{filedir1}/syy_lp_{step}.png')
        
        plt.close()
        plt.imshow(s22,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f's22_{step}')
        plt.savefig(f'{filedir1}/s22_{step}.png')
        
        plt.close()
        plt.plot((s22[nx//2,nx//2:]*160.22),'r')
        plt.grid()
        plt.title(f's22_lp_{step}')
        plt.savefig(f'{filedir1}/s22_lp_{step}.png')
        
        plt.close()
        plt.imshow(szz,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'szz_{step}')
        plt.savefig(f'{filedir1}/szz_{step}.png')
        
        plt.close()
        plt.imshow(sxy,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'sxy_{step}')
        plt.savefig(f'{filedir1}/sxy_{step}.png')
        
        plt.close()
        plt.imshow(s12,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f's12_{step}')
        plt.savefig(f'{filedir1}/s12_{step}.png')
        
        plt.close()
        plt.plot((s12[nx//2,nx//2:]*160.22),'r')
        plt.grid()
        plt.title(f's12_lp_{step}')
        plt.savefig(f'{filedir1}/s12_lp_{step}.png')
        
        plt.close()
        plt.imshow(exx,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'exx_{step}')
        plt.savefig(f'{filedir1}/exx_{step}.png')

        plt.close()
        plt.imshow(eyy,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'eyy_{step}')
        plt.savefig(f'{filedir1}/eyy_{step}.png')
        
        plt.close()
        plt.imshow(ezz,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'ezz_{step}')
        plt.savefig(f'{filedir1}/ezz_{step}.png')
        
        plt.close()
        plt.imshow(exy,origin='lower',cmap='jet')
        plt.colorbar()
        plt.title(f'exy_{step}')
        plt.savefig(f'{filedir1}/exy_{step}.png')
    
    plt.close()
    plt.imshow(ftotal,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'ftotal_{step}')
    plt.savefig(f'{filedir1}/ftotal_{step}.png')
    
    plt.close()
    plt.imshow(felas1,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'felas1_{step}')
    plt.savefig(f'{filedir1}/felas1_{step}.png')
    
    plt.close()
    plt.imshow(felas2,origin='lower',cmap='jet')
    plt.colorbar()
    plt.title(f'felas2_{step}')
    plt.savefig(f'{filedir1}/felas2_{step}.png')

print("DONE")
