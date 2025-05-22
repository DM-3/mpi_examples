import numpy as np
import cv2 as cv
import time


# initialize field
WIDTH = 300
HEIGHT = 300
field = np.where(np.random.rand(HEIGHT+2, WIDTH+2) > 0.5, 
                 np.uint8(1), 
                 np.uint8(0))


# "gameplay" loop
while cv.waitKey(1000) != 27:

    # display
    cv.imshow('field', cv.resize(field[1:-1,1:-1] * 255, 
                                    (900, 900), 
                                    interpolation=cv.INTER_NEAREST))

    # update

    start_time = time.time()

    # copy torus padding
    field[0,:] = field[-2,:]
    field[-1,:] = field[1,:]
    field[:,0] = field[:,-2]
    field[:,-1] = field[:,1]

    field[0,0] = field[-2,-2]
    field[0,-1] = field[-2,1]
    field[-1,0] = field[1,-2]
    field[-1,-1] = field[1,1]

    # count neighbors for all pixels
    w = np.asarray([[1,1,1],
                    [1,0,1],
                    [1,1,1]], dtype=np.uint8)
    count = cv.filter2D(field, -1, w)

    # apply rules
    t_field = np.zeros_like(field)
    t_field += np.where((field * count) == 2, np.uint8(1), np.uint8(0))         # alive cells with 2 neighbors
    t_field += np.where((field * count) == 3, np.uint8(1), np.uint8(0))         # alive cells with 3 neighbors
    t_field += np.where(((1 - field) * count) == 3, np.uint8(1), np.uint8(0))   # dead cells with 3 neighbors
    np.copyto(field, t_field)

    end_time = time.time()
    print(f'\r{(end_time - start_time) * 1000:.2f} ms', end=' ')
