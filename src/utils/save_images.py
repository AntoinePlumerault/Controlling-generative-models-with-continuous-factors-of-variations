from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os



def save_images(images, save_path):
    
    if isinstance(images.flatten()[0], np.floating):
        
        # [-1, 1] -> [0,255]
        if np.min(images.flatten()) < 0:            
            images = ((images + 1.) * 127.5).astype('uint8')
        # [0, 1] -> [0, 255]
        else:                                      
            images = (images * 255).astype('uint8')
    
    n_images = images.shape[0]
    n_rows = int(np.sqrt(n_images))
    while n_images % n_rows != 0:
        n_rows -= 1
    n_cols = n_images // n_rows

    # black and white images
    if images.ndim == 3: 
        h, w = images[0].shape[:2]
        images = np.expand_dims(images, -1)
        images = np.tile(images[::-1,:], [1,1,3])
        full_image = np.zeros((h * n_rows, w * n_cols, 3))
    
    # color images
    elif images.ndim == 4:
        h, w = images[0].shape[:2]
        full_image = np.zeros((h * n_rows, w * n_cols, 3))
    else:
        raise ValueError('images must be of rank 3 or 4')

    for n, image in enumerate(images):
        i = n % n_cols
        j = n // n_cols
        full_image[j * h:j * h + h, i * w:i * w + w] = image

    image_with_alpha = np.full([h * n_rows, w * n_cols, 4], 255, dtype='uint8')
    image_with_alpha[:,:,:3] = full_image[::-1,:,:]

    # Save image
    width = image_with_alpha.shape[1]
    height = image_with_alpha.shape[0]
    buffer = image_with_alpha.tobytes()

    def write_png(buffer, width, height):
        import zlib, struct
        width_byte_4 = width * 4
        raw_data = b"".join(b'\x00' + buffer[span:span + width_byte_4] for span in range((height - 1) * width * 4, -1, - width_byte_4))
        def png_pack(png_tag, data):
            chunk_head = png_tag + data
            return struct.pack("!I", len(data)) + chunk_head + struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head))

        return b"".join([
            b'\x89PNG\r\n\x1a\n',
            png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
            png_pack(b'IDAT', zlib.compress(raw_data, 9)),
            png_pack(b'IEND', b'')])
    
    with open(save_path, 'wb') as f:
        f.write(write_png(buffer, width, height))