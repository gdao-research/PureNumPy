# PureNumPy
Functions and objects help to understand concepts from computer vision and neural networks.

### Digital Image Processing Support Functions:
- conv2d(img, kernel): use Fast Fourier Transform for Convolution operation
- rgb2hsv(img): convert rgb image to hsv
- rgb2gray(img, method='avg', format='rgb'): if loaded image by opencv and method != 'avg' --> format='bgr'
- sobel(img, return_direction=False): sobel edge detection
- canny(img, k=11, sigma=1, alpha=0.1, beta=0.2, return_direction=False): canny edge detection
- make_gaussian_kernel(size, sigma): create Gaussian kernel
- dilate(img, strel): dilation of image with structure element
- erose(img, strel): erosion of image with structure element
- histeq(img): histogram equalization
- hough_circle_accumulator(edge_img, R_min=3, R_max=None, center_inside=True): return accumulator of circle detection using Hough transform
- hough_line_accumulator(edge_img): return accumulator of line detection using Hough transform
- connected_component_labeling(bw): return objects map and their sizes
- imfill(bw): flood fill algorithm

### Neural Networks:
- initializers: return an initializer function requires shape input as type of tuple
  + variance_scaling_initializer(factor=2., mode='FAN_IN', uniform=False, dtype='float32')
  + xavier_initializer(uniform=True, dtype='float32')
  + zeros_initializer(dtype='float32')
  + normal_initializer(mean=0., std=1e-3, dtype='float32')
  + uniform_initializer(lower=-1, upper=1, dtype='float32')
  + glorot_initializer(dtype='float32')

### Contact:
Please contact gdao.research@gmail.com if you have any problem using this packakge or functions/objects does not output correctly.
