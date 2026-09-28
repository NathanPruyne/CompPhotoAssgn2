Each question contains its own `.py` files. Usage for each file is provided through the argparse help text, use e.g. `python q1.py -h` to view.

I converted to `.tiff` files using `dcraw_linear.sh`, changing the input and output directories for steps. Note that I also added trailing 0s to the exposure numbers on the original `door_stack` image in order to make them sortable. 

### Question 1

All in `q1.py`. All generated photos are in `data/hdr_outs`. I found the best file to be `rendered_gaussian_linear.hdr`.

### Question 2

`q2_get_patches.py` contains a script to get patches from the color picker through an interactive interface, exported to `data/patches.npy`.

`q2_colorcorrect.py` performs color correction. Outputs are in `data/hdr_outs/colorcorrected...`.

### Question 3

`q3.py` contains all code. Resulting tonemapped outputs with my preferred burn and key are in `data/tonemap_outs`

### Question 4

My raw images are in `data/myphoto`, and the outputs of both the HDR image and after tonemapping are in `data/myphoto_outs`.

### Question 5

`capture_noise.sh` was used to automatically take the photos with the lens cap on and of the ramp.

`q5_darkframe.py` contains calculations of the dark frame, outputting plots to `../data/pixel_histogram.png` and `../data/mean_variance.png`. The new weighting scheme is added to `q1.py`.