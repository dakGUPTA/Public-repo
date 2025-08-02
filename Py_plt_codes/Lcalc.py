import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(file_path):
    return np.loadtxt(file_path)

step = int(input("Enter step value: "))
nx=ny=256
#data = np.loadtxt(f"prof_gp.{step}")
#comp = data[:,2].reshape(nx,ny)
#plt.plot(comp[:,ny//2], '-o')
#plt.title(f'Comp_1D_{step}')
#plt.show()

root_dir = './'
file_name = f'prof_gp.{step}'
Dc = np.array([], dtype=float)
L = np.array([], dtype=float)
c_eq = 0.15
ds = '\u0394'
#dir_val = ["0.01","0.02","0.03","0.04","0.05","0.06", "0.07", "0.08", "0.09","0.1"]
dir_val = ["0.001", "0.01","0.1", "1", "10"]
#for i in range(5, 105, 5):
for i in dir_val:
    #subdir_name = f'L_{i/100:.2f}'.rstrip('0').rstrip('.')
    subdir_name = f"L_{i}"
    print(f'Processing Subdir: {subdir_name}')
    file_path = os.path.join(root_dir, subdir_name, file_name)
    
    try:
        data = load_data(file_path)
        comp = data[:,2].reshape(nx,ny)
        min_comp = np.min (comp)
        dc = (min_comp - c_eq)
        print(rf'{ds}c : {dc}')
        Dc = np.append(Dc, dc)
        #L = np.append(L, (i/100))
        L = np.append(L, i)
        
    except Exception as e:
        print(f'Error loading file {file_path}: {e}')

plt.plot(L,Dc, '-o')
plt.title("Dc vs L")
plt.savefig(f"DcvsL_{nx}_{step}.png")
#plt.show()
sys.exit()
y = np.array(Dc, dtype=float)
x = np.array(L, dtype=float)

coefficients = np.polyfit(x, y, 3)
polynomial = np.poly1d(coefficients)

x_fit = np.linspace(min(x), max(x), 100)
y_fit = polynomial(x_fit)

# Plot the data and the fit line
plt.scatter(x, y, label=f'Actual_data')
plt.plot(x_fit, y_fit, label='Fit Line', color='red')
plt.title(rf'Polynomial fit (degree = 3)')
plt.xlabel(r'$\Delta c$')
plt.ylabel('L')
plt.legend()
plt.savefig(f'{nx}_{step}_Lfit.png')
plt.show()

# Extrapolate to a new value
x_new = 0.0000000 
y_new = polynomial(x_new)
print(f'Extrapolated value at x={x_new}: y={y_new}')
