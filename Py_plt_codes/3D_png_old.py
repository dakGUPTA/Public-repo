import os
import vtk
import pyvista as pv
from pyvista import wrap

# Start off-screen rendering (important for headless servers)
pv.start_xvfb()

# Output folder for images
output_folder = "3D_Images"
os.makedirs(output_folder, exist_ok=True)

# 🔍 Scalar field to use
scalar_name = "COMP"  # Replace with your actual scalar name
iso_value = 0.5       # �� Isovalue for isosurface extraction

# Camera views
angles = {
    'xy': 'xy',
    'xz': 'xz',
    'yz': 'yz',
    'iso': 'iso',
}

# Step input
#step = int(input("Enter the step value: "))
#file_type = input("Enter the file_type (vtk or h5): ") 
step = 0
file_type = "h5"

# Determine filename
if file_type == "vtk":
    file_name = f"DATA/Output_{step:07d}.vtk"
elif file_type == "h5":
    file_name = f"bin/DATA/3dkks_{step}.xdmf"
else:
    print("Unsupported file type.")
    exit()

print(f"📂 Processing: {file_name}")

# Read the mesh
if file_type == "vtk":
    mesh = pv.read(file_name)
elif file_type == "h5":
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(file_name)
    reader.Update()
    mesh = wrap(reader.GetOutputDataObject(0))

# Check if scalar is present
if scalar_name not in mesh.array_names:
    print(f"❌ Scalar '{scalar_name}' not found in {file_name}.")
    print(f"   Available scalars: {mesh.array_names}")
    exit()

# Generate isosurface
isosurface = mesh.contour(isosurfaces=[iso_value], scalars=scalar_name)

# Save screenshots for each angle
for angle_name, cam_pos in angles.items():
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.add_mesh(isosurface, color="tomato", opacity=1.0, show_edges=False)
    plotter.camera_position = cam_pos
    img_filename = f"{angle_name}_{step:07d}.png"
    img_path = os.path.join(output_folder, img_filename)
    plotter.screenshot(img_path)
    plotter.close()
    print(f"✅ Saved: {img_path}")

