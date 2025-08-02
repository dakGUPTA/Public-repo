import os
import subprocess
import shutil
import vtk
import pyvista as pv
from pyvista import wrap
from glob import glob
pv.start_xvfb()  

def create_movie(angle_key, step):
    input_files = sorted(glob(os.path.join(output_folder, f"{angle_key}_*.png")))
    filtered_files = input_files[::step]
    if not filtered_files:
        print(f"⚠️ No images found for angle '{angle_key}' with step {step}")
        return

    # Create temp directory
    temp_dir = os.path.join(output_folder, f"temp_{angle_key}_step{step}")
    os.makedirs(temp_dir, exist_ok=True)

    # Copy every nth frame
    for i, src_file in enumerate(filtered_files):
        dst_file = os.path.join(temp_dir, f"{i:07d}.png")
        shutil.copyfile(src_file, dst_file)

    # Create movie
    movie_filename = f"{angle_key}_isosurface.mp4"
    output_path = os.path.join(output_folder, movie_filename)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(frame_rate),
        "-i", os.path.join(temp_dir, "%07d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    print(f"🎬 Generating movie for '{angle_key}' with step {step}...")
    subprocess.run(ffmpeg_cmd)
    print(f"✅ Movie saved to: {output_path}")

    # Clean up
    shutil.rmtree(temp_dir)

####======> Main <========####
# Output folder for images
output_folder = "vtk_isosurfaces"
os.makedirs(output_folder, exist_ok=True)

# 🔍 Scalar field to use
#scalar_name = "PHI0"  # Replace with your actual scalar name
scalar_name = "beta"  
iso_value = 0.5       # 🎯 The isovalue at which to extract the surface

# Camera views
angles = {'xy': 'xy', 'xz': 'xz', 'yz': 'yz','iso': 'iso',}
#angles = {'iso': 'iso',}

# Input
plot_images = int(input("Enter 1 for plotting images else 0: "))
make_movie = int(input("Enter 1 for making movie else 0: "))
if (make_movie == 1):
    view_angle = input("Enter which view angle to make movie (iso, xy,yz,xz): ")
file_type = input("Enter vtk for .vtk format and h5 for .h5 format: ")

# 🔁 Step ranges
step_values = []
step_values += list(range(0, 300000 + 1, 2000))
#step_values += list(range(10000, 190000 + 1, 10000))
#step_values += list(range(20000, 100000 + 1, 10000))
print(step_values)

frame_rate = 2  # For the output video
step_movie_values = [1] 

# 📸 Plot images
if plot_images == 1:
    for step in step_values:
        
        if file_type == "vtk":
            file_name = f"DATA/Output_{step:07d}.vtk"

            if not os.path.exists(file_name):
                print(f"File not found: {file_name}")
                continue
        elif file_type == "h5":
            file_name = f"bin/3dkks_{step}.xdmf"

            if not os.path.exists(file_name):
                print(f"File not found: {file_name}")
                continue

        print(f"Processing: {file_name}")
        if file_type == "vtk":
            mesh = pv.read(file_name)
        elif file_type == "h5":
            reader = vtk.vtkXdmfReader()
            reader.SetFileName(file_name)
            reader.Update()
            mesh = wrap(reader.GetOutputDataObject(0))


        if scalar_name not in mesh.array_names:
            print(f"Scalar '{scalar_name}' not found in {file_name}. Available scalars: {mesh.array_names}")
            continue

        # Generate isosurface
        isosurface = mesh.contour(isosurfaces=[iso_value], scalars=scalar_name)

        # Save screenshot for each angle
        for angle_name, cam_pos in angles.items():
            plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
            plotter.add_mesh(isosurface, color="tomato", opacity=1.0, show_edges=False)
            plotter.camera_position = cam_pos
            img_filename = f"{angle_name}_{step:07d}.png"
            img_path = os.path.join(output_folder, img_filename)
            plotter.screenshot(img_path)
            plotter.close()
            print(f"✅ Saved: {img_path}")

# 🎞️ Movie creation
if make_movie == 1:
    if (view_angle == "all"):
        for angle_key in angles.keys():
            for step in step_movie_values:
                create_movie(angle_key, step)
    else:
        for step in step_movie_values:
            create_movie(view_angle, step)


