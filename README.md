# InteractiveDenoiserWithAffinity
Implementation for Interactive Monte Carlo Denoising using Affinity of Neural Features

# How to use
```
./trainer/denoise.py model_file last_layer_size input_director output_file_name
Example: ./trainer/denoise.py ./11Full1NewDBmodel.pth 11 "./staircase reference" ./staircase.hdr
```