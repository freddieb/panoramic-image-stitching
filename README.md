# Panoramic Image Stitching

Panoramic image stitcher based on ['Automatic Panoramic Image Stitching using Invariant Features'](http://matthewalunbrown.com/papers/ijcv2007.pdf) by Matthew Brown and David Lowe.

## Installation
Using Python 3, install:
- numpy
- cv2
- argparse
- scipy
- ordered_set

## Run
```
python3 ./src/stitcher/main.py <path_to_input_images_directory>
```

## Method Overview
1. Extract SIFT features from all images
2. Find similar SIFT features using KD tree
3. Verify matches using RANSAC, more inliers = good match
4. Iteratively add each image to the bundle adjuster (with the best matches being added first)

### Bundle Adjustment Notes
- Rotation matrix is converted to rotation vectors to allow the rotation to be parameterised with fewer variables in order to reduce the problem size
- Uses Symbolic differentiation to build the Jacobian matrix
- Optimises for the reprojection error between matches, this bundle adjuster does not take into account the 3D projections however this will be added as an option in the future

## Datasets
- The Adobe Panoramas Dataset has been used for testing: https://sourceforge.net/adobe/adobedatasets/panoramas/home/Home/