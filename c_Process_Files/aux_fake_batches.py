import os
from glob import glob

dr = r'd:\DATA\PTD\new\tuberculosis\v2.5\img_previews'

i = 0
mask_files = glob(dr + '/*-mask.png')
for mask_file in mask_files:
    im_file = mask_file.replace('-mask.png', '.png')
    print(im_file)

    batch_filename = os.path.join(dr, 'batch%06i.txt' % i)
    with open(batch_filename, 'wt') as f:
        f.write('filenames\n' + os.path.split(im_file)[1] + '\n')

    os.rename(im_file, batch_filename.replace('.txt', '.png'))
    os.rename(mask_file, batch_filename.replace('.txt', '-mask.png'))

    i += 1
