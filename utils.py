import numpy as np

palletes = {
    0: [0,0,0],
    1: [255,0,0],
    2: [255,255,0],  
    3: [0,255,0],
    4: [0,255,255]
}
# 3: [255,0,255],
def mask2rgb(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    for i in np.unique(mask):
            rgb[mask==i] = palletes[i]
            
    return rgb

def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k,v in palletes.items():
        mask[np.all(rgb==v, axis=2)] = k
        
    return mask