import os
import vtk
import pyvista as pv
from pyvista import wrap

# Off-screen rendering (ignore deprecation warning for now)
pv.start_xvfb()

# Output folder
output_folder = "3D_Images"
os.makedirs(output_folder, exist_ok=True)

# Input parameters
scalar_name = "PHI0"
iso_value = 0.5
step = int(input("Enter step value: "))
file_type = "h5"

# Camera angles
angles = {
    'xy': 'xy',
    'xz': 'xz',
    'yz': 'yz',
    'iso': 'iso',
}

# File path
if file_type == "vtk":
    file_name = f"DATA/Output_{step:07d}.vtk"
elif file_type == "h5":
    file_name = f"mpkks_{step}.xdmf"
else:
    print("Unsupported file type.")
    exit()

print(f"📂 Processing: {file_name}")

# Read mesh
if file_type == "vtk":
    mesh = pv.read(file_name)
elif file_type == "h5":
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(file_name)
    reader.Update()
    mesh = wrap(reader.GetOutputDataObject(0))

print(mesh)

# Check scalar field
if scalar_name not in mesh.array_names:
    print(f"❌ Scalar '{scalar_name}' not found.")
    print("   Available scalars:", mesh.array_names)
    exit()

# Check scalar range
arr = mesh.point_data[scalar_name]
scalar_min, scalar_max = arr.min(), arr.max()
print(f"ℹ️ Scalar '{scalar_name}' range: min={scalar_min}, max={scalar_max}")

if not (scalar_min <= iso_value <= scalar_max):
    print(f"⚠️ iso_value={iso_value} is outside scalar range.")
    exit()

# Generate isosurface
isosurface = mesh.contour(isosurfaces=[iso_value], scalars=scalar_name)

# Handle empty isosurface
if isosurface.n_points == 0:
    print("❌ Empty isosurface. No mesh to plot.")
    exit()

# Plot and save screenshots
for angle_name, cam_pos in angles.items():
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.add_mesh(isosurface, color="tomato", opacity=1.0, show_edges=False)
    plotter.camera_position = cam_pos
    img_filename = f"{angle_name}_{step:07d}.png"
    img_path = os.path.join(output_folder, img_filename)
    plotter.screenshot(img_path)
    plotter.close()
    print(f"✅ Saved: {img_path}")

