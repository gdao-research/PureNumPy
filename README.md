# PureNumPy
Functions and objects help to understand concepts from computer vision and neural networks.

### Digital Image Processing
- conv2d(img, kernel): use Fast Fourier Transform for Convolution operation
- rgb2hsv(img): return hsv image
- rgb2gray(img, method='avg', format='rgb'): return gray image
- sobel(img, return_direction=False): return sobel edge detection
- canny(img, k=11, sigma=1, alpha=0.1, beta=0.2, return_direction=False): return canny edge detection
- make_gaussian_kernel(size, sigma): return Gaussian kernel
- dilate(img, strel): return dilated image with structure element
- erose(img, strel): return erosed image with structure element
- histeq(img): return equalized image
- hough_circle_accumulator(edge_img, R_min=3, R_max=None, center_inside=True): return accumulator of circle detection using Hough transform
- hough_line_accumulator(edge_img): return accumulator of line detection using Hough transform
- connected_component_labeling(bw): return objects map and their sizes
- imfill(bw): return objects filled image
- hog_feature(img): return Histogram of Gradient feature
- harris_corner_detector(img, threshold, kernel_size=3, p=0.5): return corner indices

### Neural Networks
- initializers: return an initializer function requires shape input as type of tuple
  + variance_scaling_initializer
  + xavier_initializer
  + zeros_initializer
  + normal_initializer
  + uniform_initializer
  + glorot_initializer
- layers: return a layer class
  + Conv2D
  + MaxPooling2D
  + Dense
  + ReLU
  + PReLU
- losses: return a loss function and its derivative
  + softmax
  + mean_squared_error
- optimizers: return an optimizer class
  + SGD
  + SGDMomentum
  + RMSProp
  + Adam

### How to Use
- Image processing package contains functions, please check the code to understand how the related functions work.

- Neural networks package is more object oriented and contains supportive objects. Check out [test_nn.py](./test_nn.py) to setup training process.

This project is developed and managed during my PhD study at the [UNC Charlotte](https://www.uncc.edu/) in my spare time.
