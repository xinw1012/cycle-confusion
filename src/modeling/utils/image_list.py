import numpy as np

def crop_tensor(image, crop_sizes, startx=None, starty=None):
    image = image.clone()
    indx = image.size(-2) - crop_sizes[0]
    indy = image.size(-1) - crop_sizes[1]
    if startx is None:
        if indx == 0:
            startx = 0
        else:
            startx = np.random.choice(indx)
    if starty is None:
        if indy == 0:
            starty = 0
        else:
            starty = np.random.choice(indy)
    return image[:, startx:startx+crop_sizes[0],
           starty:starty+crop_sizes[1]], startx, starty
