# Process Files

Working with data files and image directories, not with any `*.sqlite` files. 

* `aux_make_whole_lungs_dataset.py` composes data sets using old version of lung segmentation
(depricated)
* `run_a_prepare_previews.py` reads the original full-size 16-bit PNGs and makes their 
8-bit preview versions of size `256x256` ready for lung segmentation 
* `run_b_segment_lungs_*.py` run lung segmentation on the 8-bit previews and creates 
`*-mask.png` files with lung masks 
* `run_c_prepare_dataset.py` reads the original full-size 16-bit PNGs, applies the pre-calculated
lung masks, performs normalization using lung pixels intensities, 
performs cropping (when `to_crop = True`),
performs static image augmentation (when `to_augment = True`)
and creates the corresponding files
