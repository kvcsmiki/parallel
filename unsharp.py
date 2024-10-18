#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:57:52 2024

@author: mac
"""

import cv2
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


# Function to process one tile
def process_tile(args):
    image, start_row, end_row = args
    blurred = cv2.GaussianBlur(image[start_row:end_row], (9, 9), 0)
    high_pass = cv2.subtract(image[start_row:end_row], blurred)
    sharpened = cv2.add(image[start_row:end_row], 3 * high_pass)
    return sharpened

def unsharp_masking_cpu(image, num_processes=4):
    height = image.shape[0]
    tile_height = height // num_processes

    args = [(image, i * tile_height, (i + 1) * tile_height) for i in range(num_processes)]

    with Pool(num_processes) as pool:
        result_tiles = pool.map(process_tile, args)

    output = np.vstack(result_tiles)
    return output

def unsharp_masking_simd(image, num_threads=4):
    height = image.shape[0]
    tile_height = height // num_threads

    args = [(image, i * tile_height, (i + 1) * tile_height) for i in range(num_threads)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        result_tiles = list(executor.map(process_tile, args))

    output = np.vstack(result_tiles)
    return output


if __name__ == '__main__':
    image = cv2.imread('input.jpg')
    if image is None:
        print("Couldn't load image.")
    else:
        cpu_sharpened = unsharp_masking_cpu(image)
        cv2.imwrite('cpu.jpg', cpu_sharpened)
        
        simd_sharpened = unsharp_masking_simd(image)
        cv2.imwrite('simd.jpg', simd_sharpened)
