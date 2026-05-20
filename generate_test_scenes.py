import os
import math

# List all the accelerators you want to benchmark
ACCELERATORS = ["bvh", "kdtree", "uniformgrid", "twolevelgrid"]

def get_header(accelerator, filename):
    return f"""
LookAt 0 20 40  0 0 0  0 1 0
Camera "perspective" "float fov" [45]
Film "rgb" "integer xresolution" [512] "integer yresolution" [512] "string filename" ["{filename}.exr"]
Sampler "halton" "integer pixelsamples" [4]
Integrator "path" "integer maxdepth" [2]
Accelerator "{accelerator}"

WorldBegin
LightSource "distant" "point3 from" [10 20 10] "point3 to" [0 0 0] "rgb L" [3 3 3]
AttributeBegin
  Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
"""

footer = "\nAttributeEnd\n"

def generate_lattice(scale=1.0, spacing=2.0, count=12):
    geom = ""
    offset = (count * spacing) / 2.0
    for x in range(count):
        for y in range(count):
            for z in range(count):
                px = (x * spacing - offset) * scale
                py = (y * spacing - offset) * scale
                pz = (z * spacing - offset) * scale
                rad = 0.8 * scale

                geom += '  AttributeBegin\n'
                geom += f'    Translate {px} {py} {pz}\n'
                geom += f'    Shape "sphere" "float radius" [{rad}]\n'
                geom += '  AttributeEnd\n'
                
    return geom

os.makedirs("test_scenes", exist_ok=True)

# Generate a file for every combination of scene and accelerator
for accel in ACCELERATORS:
    
    # 1. SMALL DIFFERENCE
    with open(f"test_scenes/test_small_{accel}.pbrt", "w") as f:
        f.write(get_header(accel, "small"))
        f.write(generate_lattice())
        f.write(footer)

    # 2. MEDIUM DIFFERENCE
    with open(f"test_scenes/test_medium_{accel}.pbrt", "w") as f:
        f.write(get_header(accel, "medium"))
        f.write(generate_lattice())
        f.write("""
  AttributeBegin
    Material "diffuse" "rgb reflectance" [0.2 0.2 0.8]
    Shape "trianglemesh" "integer indices" [0 1 2 0 2 3] 
        "point3 P" [-500 -12 -500   500 -12 -500   500 -12 500   -500 -12 500]
  AttributeEnd
""")
        f.write(footer)

    # 3. HIGH DIFFERENCE
    with open(f"test_scenes/test_high_{accel}.pbrt", "w") as f:
        f.write(get_header(accel, "high"))
        f.write(generate_lattice(scale=0.05, spacing=1.0, count=20))
        f.write("""
  AttributeBegin
    Shape "trianglemesh" "integer indices" [0 1 2] 
        "point3 P" [-1000 -1000 -1000   1000 -1000 -1000   0 1000 -1000]
  AttributeEnd
""")
        f.write(footer)

print(f"Generated 3 test scenes for each of the {len(ACCELERATORS)} accelerators!")