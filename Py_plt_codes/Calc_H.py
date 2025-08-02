import numpy as np
import matplotlib.pyplot as plt

step = int(input("Enter the Step value: "))
nx = ny = 512
Z_component = int(input("Enter 1 for using 33 components else 0: "))

data = np.loadtxt(f"prof_gp.{step}")

s11 = data[:,8].reshape(nx,ny)
s22 = data[:,9].reshape(nx,ny)

e11 = data[:,12].reshape(nx,ny)
e22 = data[:,13].reshape(nx,ny)

if (Z_component == 1):
    s33 = data[:,10].reshape(nx,ny)
    s12 = data[:,11].reshape(nx,ny)
    
    e33 = data[:,14].reshape(nx,ny)
    e12 = data[:,15].reshape(nx,ny)
else:
    s12 = data[:,10].reshape(nx,ny)
    e12 = data[:,13].reshape(nx,ny)


sts_mat = np.zeros((nx,ny,3,3))
str_mat = np.zeros((nx,ny,3,3))

Eigvals = np.zeros((nx,ny,3))
Eigvals1 = np.zeros((nx,ny,3))

## Principal stress & strains.
Sig1 = np.zeros((nx,ny))
Sig2 = np.zeros((nx,ny))
Sig3 = np.zeros((nx,ny))
Eps1 = np.zeros((nx,ny))
Eps2 = np.zeros((nx,ny))
Eps3 = np.zeros((nx,ny))

## Hydrostatic stress & strains
Hsts = np.zeros((nx,ny))
Hstr = np.zeros((nx,ny))

for i in range (nx):
    for j in range(ny):

        sts_mat[i,j,0,0] = s11[i,j]
        sts_mat[i,j,0,1] = s12[i,j]
        sts_mat[i,j,0,2] = 0.0 
        sts_mat[i,j,1,0] = s12[i,j]
        sts_mat[i,j,1,1] = s22[i,j]
        sts_mat[i,j,1,2] = 0.0 
        sts_mat[i,j,2,0] = 0.0 
        sts_mat[i,j,2,1] = 0.0 
        if (Z_component == 1):
            sts_mat[i,j,2,2] = s33[i,j] 

        str_mat[i,j,0,0] = e11[i,j]
        str_mat[i,j,0,1] = e12[i,j]
        str_mat[i,j,0,2] = 0.0 
        str_mat[i,j,1,0] = e12[i,j]
        str_mat[i,j,1,1] = e22[i,j]
        str_mat[i,j,1,2] = 0.0 
        str_mat[i,j,2,0] = 0.0 
        str_mat[i,j,2,1] = 0.0 
        if (Z_component == 1):
            str_mat[i,j,2,2] = e33[i,j] 
        
        Eigvals[i,j] = np.linalg.eigvals(sts_mat[i,j])
        Sig1[i,j] = Eigvals[i,j,0]
        Sig2[i,j] = Eigvals[i,j,1]
        Sig3[i,j] = Eigvals[i,j,2]

        Eigvals1[i,j] = np.linalg.eigvals(str_mat[i,j])
        Eps1[i,j] = Eigvals1[i,j,0]
        Eps2[i,j] = Eigvals1[i,j,1]
        Eps3[i,j] = Eigvals1[i,j,2]
        
        Hsts[i,j] = np.mean(Eigvals[i,j])
        Hstr[i,j] = np.mean(Eigvals1[i,j])

plt.imshow(Hsts,cmap = 'jet')
plt.colorbar()
plt.title(f"Hydrostatic stress_{step}")
plt.savefig(f"./Pngs/hsts_{step}")
plt.show()

plt.imshow(Hstr,cmap = 'jet')
plt.colorbar()
plt.title(f"Hydrostatic strain_{step}")
plt.savefig(f"./Pngs/hstr_{step}")
plt.show()

