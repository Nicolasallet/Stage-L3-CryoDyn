#!/usr/bin/env python
import cv2
import skimage.morphology
from plantcv import plantcv as pcv
import numpy as np
import sys
import os

for arg in sys.argv:
    print(arg)


img = cv2.imread(arg,0)

biimg=img!=0
biimg=skimage.morphology.skeletonize(biimg)

img2=np.array(biimg*255,dtype=np.uint8)

ll=np.max(np.shape(img2))

pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=img2, size=ll)

cv2.imwrite('pruned_'+arg, pruned_skeleton)
