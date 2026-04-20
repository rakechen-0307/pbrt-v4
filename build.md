# Build Guide For GPU
## Requirements
1. CMake
2. nvcc
3. GCC/G++ 12
4. [NVIDIA OptiX SDK 7.7](https://developer.nvidia.com/designworks/optix/downloads/legacy)

## Build Commands
```bash
mkdir build
cmake -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12 -DCMAKE_BUILD_TYPE=Release -DPBRT_OPTIX_PATH=<your_path>/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64 -DCMAKE_CUDA_ARCHITECTURES=89 -DPBRT_GPU_SHADER_MODEL=sm_89 -DPBRT_FLOAT_AS_DOUBLE=OFF -B build
cmake --build build
```

## Render Command Example
```bash
./build/pbrt --gpu pbrt-v4-scenes/barcelona-pavilion/pavilion-day.pbrt

# Convert .exr to .png
./build/imgtool convert --outfile ./pavilion-day.png --colorspace srgb --aces-filmic ./pavilion-day.exr
```