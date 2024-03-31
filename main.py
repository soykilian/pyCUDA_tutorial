import argparse
import sys
import timeit

import numpy as np
import PIL.Image

import math

import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler
import gray_cpu

import numpy as np

DIM_BLOCK = 32
def apply_gpu(image_src):
    image_dst = np.empty_like(image_src)
    red_channel = image_src[:, :, 0].copy()
    green_channel = image_src[:, :, 1].copy()
    blue_channel = image_src[:, :, 2].copy()

    height, width = image_src.shape[:2]

    dim_grid_x = math.ceil(width / DIM_BLOCK)
    dim_grid_y = math.ceil(height /DIM_BLOCK)

    max_num_blocks = (
            pycuda.autoinit.device.get_attribute(
                driver.device_attribute.MAX_GRID_DIM_X
            )
            * pycuda.autoinit.device.get_attribute(
                driver.device_attribute.MAX_GRID_DIM_Y
            )
    )

    if (dim_grid_x * dim_grid_y) > max_num_blocks:
        raise ValueError(
            'image dimensions too great, maximum block number exceeded'
        )

    mod = compiler.SourceModule(open('./imgray.cu').read())
    apply_filter = mod.get_function('applyFilter')

    apply_filter(
        driver.InOut(red_channel),
        driver.InOut(green_channel),
        driver.InOut(blue_channel),
        np.uint32(width),
        np.uint32(height),
        block=(DIM_BLOCK, DIM_BLOCK, 1),
        grid=(dim_grid_x, dim_grid_y)
    )

    image_dst[:, :, 0] = red_channel
    image_dst[:, :, 1] = green_channel
    image_dst[:, :, 2] = blue_channel

    return image_dst

def main():
    program_start = timeit.default_timer()
    parser = argparse.ArgumentParser(description='apply grayscale or gaussian blur filter to PNG image')
    parser.add_argument('image_src', help='source image file path')
    parser.add_argument('image_result', help='resulting image file path')
    parser.add_argument('-g', help='use GPU for processing', action='store_true')
    args = parser.parse_args()
    image_dst = None
    try:
        image = PIL.Image.open(args.image_src)
        image_src = np.array(image)  # shape = (height, width, channels)
    except FileNotFoundError as e:
        sys.exit(e)
    if args.g:
        print("Excecuting GPU version")
        image_dst = apply_gpu(image_src)
    else:
        print("Excecuting CPU version")
        image_dst = gray_cpu.apply_cpu(image_src)
    try:
        save_start = timeit.default_timer()
        PIL.Image.fromarray(image_dst).save(args.image_result)
        save_end = timeit.default_timer()
        print('Time spent saving image:', save_end - save_start, 'seconds')
    except OSError as e:
        print(e, file=sys.stderr)
    program_end = timeit.default_timer()
    print('Total time spent running:', program_end - program_start, 'seconds')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
