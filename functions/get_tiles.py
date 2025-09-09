import pyvips
import numpy as np
import cv2
import pathlib

def get_tiles(input_path, output_path):
    blank_ratio  = 0.95     
    s_thresh = 30          
    v_thresh = 230          
    tile_size = 2048      


    image = pyvips.Image.new_from_file(input_path, access='sequential')
    width, height = image.width, image.height
    bands = image.bands

    tile_id = 0
    kept_tiles = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            tile = image.crop(x, y, w, h)

            np_tile_raw = np.ndarray(
                buffer=tile.write_to_memory(),
                dtype=np.uint8,
                shape=[tile.height, tile.width, bands]
            )

 
            hsv = cv2.cvtColor(np_tile_raw, cv2.COLOR_RGB2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            tissue_mask = (s > s_thresh) & (v < v_thresh)
            tissue_ratio = np.sum(tissue_mask) / tissue_mask.size

            if tissue_ratio > (1 - blank_ratio):
                if w != tile_size or h != tile_size:
                    bg = pyvips.Image.black(tile_size, tile_size).copy(interpretation='srgb') + 255
                    if bands == 3:
                        bg = bg.bandjoin([bg, bg])
                    tile = bg.insert(tile, 0, 0)

                tile_path = output_path / f'tile_{tile_id:04d}.jpg'
                tile.jpegsave(str(tile_path), Q=90)
                tile_id += 1
                kept_tiles += 1

    print(f"✅ save {kept_tiles} tiles")
