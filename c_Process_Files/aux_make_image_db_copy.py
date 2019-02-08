import os
import shutil
from skimage import io
from imutils import imresize


def copy_all(src_dir, dst_dir, img_size):
    print('Scanning dir')
    items = os.listdir(src_dir)

    for item in items:
        item_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        if os.path.isfile(item_path):
            print(item_path)

            try:
                if item_path.endswith('.png'):
                    im = io.imread(item_path)
                    im = imresize(im, (img_size, img_size))
                    io.imsave(dst_path, im)
                else:
                    shutil.copy(item_path, dst_path)
            except:
                print('Failed to process file ' + item)
        elif os.path.isdir(item_path):
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

            copy_all(item_path, dst_path, img_size)


def main():
    img_size = 512

    from_tos = {r'e:\PTD1_PNG_PART1_1--785': r'd:\IMG\Xray_PTD_Copy512\PTD1_PNG_PART1_1--785',
                r'e:\PTD2_PNG_ALL_26--464': r'd:\IMG\Xray_PTD_Copy512\PTD2_PNG_ALL_26--464',
                r'f:\PTD1_PNG_PART2_786--1715': r'd:\IMG\Xray_PTD_Copy512\PTD1_PNG_PART2_786--1715'}

    io.use_plugin('freeimage')

    for src_dir in from_tos:
        dst_dir = from_tos[src_dir]

        copy_all(src_dir, dst_dir, img_size)


if __name__ == '__main__':
    main()

