# InteractiveDenoiserWithAffinity
Implementation for Interactive Monte Carlo Denoising using Affinity of Neural Features

# How to use denoiser
```
./trainer/denoise.py model_file last_layer_size input_director output_file_name
Example: ./trainer/denoise.py ./kernel11.pth 11 "./staircase reference" ./staircase.hdr
```
# How to use renderer
```
./PathTracer.exe base_scene_file extra_models_directory extra_textures_directory output_directory (--visualize)
Example: ./PathTracer.exe "./bathroom/scene.pbrt" "./models" "./textures" "./training images" --visualize
```