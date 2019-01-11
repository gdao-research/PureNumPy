import numpy as np


def conv2d(img, kernel, padding='valid'):
    assert img.ndim == 2, 'Image needs to be in 2d array'
    assert kernel.ndim == 2, 'Kernel needs to be in 2d array'
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1, 'Please make odd kernel size'
    if img.dtype == 'uint8':
        img = img/255

    s1 = np.array(img.shape) + np.array(kernel.shape) - 1
    fsize = 2**np.ceil(np.log2(s1)).astype('int32')
    fslice = tuple([slice(0, int(sz)) for sz in s1])
    new_x = np.fft.fft2(img, fsize)
    new_y = np.fft.fft2(kernel, fsize)
    ret = np.fft.ifft2(new_x*new_y)[fslice]
    ret = ret.real
    if padding == 'full':
        return ret
    elif padding == 'same':
        p = (kernel - 1)//2
    else:  # 'valid'
        p = kernel - 1
    return ret[p:-p, p:-p]

def rgb2hsv(img):
    assert img.ndim == 3, 'Image needs to be in 3d'
    if img.dtype == 'uint8':
        img = img/255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.max(img, axis=2)
    mn = np.min(img, axis=2)
    df = mx - mn + 1e-7
    hsv = np.zeros_like(img)
    # H
    idx = np.where(mx == mn)
    hsv[idx[0], idx[1], 0] = 0
    idx = np.where(mx == r)
    hsv[idx[0], idx[1], 0] = (60*((g[idx[0], idx[1]] - b[idx[0], idx[1]])/df[idx[0], idx[1]]) + 360).astype('int32') % 360
    idx = np.where(mx == g)
    hsv[idx[0], idx[1], 0] = (60*((b[idx[0], idx[1]] - r[idx[0], idx[1]])/df[idx[0], idx[1]]) + 480).astype('int32') % 360
    idx = np.where(mx == b)
    hsv[idx[0], idx[1], 0] = (60*((r[idx[0], idx[1]] - g[idx[0], idx[1]])/df[idx[0], idx[1]]) + 600).astype('int32') % 360
    # S
    idx = np.where(mx == 0)
    hsv[idx[0], idx[1], 1] = 0
    idx = np.where(mx != 0)
    hsv[idx[0], idx[1], 1] = df[idx[0], idx[1]]/mx[idx[0], idx[1]]
    # V
    hsv[:, :, 2] = mx
    return hsv


def rgb2gray(img, method='avg', format='rgb'):
    # format exists because cv2 load image in bgr order
    assert img.ndim == 3, 'Image needs to be in 3d'
    if img.dtype == 'uint8':
        img = img/255.0

    if method == 'avg':
        return np.mean(img, axis=2)
    else:
        R = 0.299
        G = 0.587
        B = 0.114
        return np.dot(img[..., :3], [R, G, B]) if format == 'rgb' else np.dot(img[..., :3], [B, G, R])


def sobel(img, return_direction=False):
    Kx = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = conv2d(img, Kx)
    Gy = conv2d(img, Ky)
    Gm = np.sqrt(Gx**2, Gy**2)
    if return_direction:
        return Gm, np.arctan2(Gy, Gx)
    else:
        return Gm


def make_gaussian_kernel(size, sigma):
    ax = np.arange(-size//2+1, size//2+1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2.*(sigma**2)))
    return kernel/kernel.sum()


def canny(img, k=11, sigma=1, alpha=0.1, beta=0.2, return_direction=False):
    if img.ndim == 3:
        img = rgb2gray(img)
    Kg = make_gaussian_kernel(k, sigma)
    img = conv2d(img, Kg)

    Gm, Gd = sobel(img, return_direction=True)
    Z = non_max_suspression(Gm, Gd, alpha, beta)

    T = alpha*np.max(Gm)
    t = beta*T
    edge_img = np.zeros_like(Gm, dtype='uint8')
    edge_img[Z > T] = 255
    temp1 = t < Z
    temp2 = Z < T
    temp = (temp1 * temp2).astype('bool')
    edge_img[temp] = 50
    edge = edge_linking(edge_img, 50, 255)
    if return_direction:
        return (edge == 255).astype('float32'), Gd
    else:
        return (edge == 255).astype('float32')


def edge_linking(x, t, T):
    strong = np.argwhere(x == T).tolist()
    while strong:
        r, c = strong.pop()
        temp = x[r-1:r+1, c-1:c+1]
        idx = np.argwhere(temp == t)
        if idx.size > 0:
            indices = np.asarray([r, c]) - 1 + idx
            for r, c in indices:
                x[r, c] = T
                strong.append([r, c])
    return x


def non_max_suspression(Gm, Gd, alpha, beta):
    R, C = Gm.shape
    Gm[[0, R-1], :] = 0
    Gm[:, [0, C-1]] = 0
    Z = np.zeros_like(Gm)
    edges = np.argwhere(Gm > alpha*beta*np.max(Gm))
    for edgeR, edgeC in edges:
        angle = np.rad2deg(Gd[edgeR, edgeC]) % 180
        if (0 <= angle < 22.5) or (157.5 <= angle < 180):  # angle 0
            if Gm[edgeR, edgeC] >= Gm[edgeR, edgeC-1] and Gm[edgeR, edgeC] >= Gm[edgeR, edgeC+1]:
                Z[edgeR, edgeC] = Gm[edgeR, edgeC]
        elif (22.5 <= angle < 67.5):  # angle 45
            if Gm[edgeR, edgeC] >= Gm[edgeR-1, edgeC+1] and Gm[edgeR, edgeC] >= Gm[edgeR+1, edgeC-1]:
                Z[edgeR, edgeC] = Gm[edgeR, edgeC]
        elif (67.5 <= angle < 112.5):  # angle 90
            if Gm[edgeR, edgeC] >= Gm[edgeR-1, edgeC] and Gm[edgeR, edgeC] >= Gm[edgeR+1, edgeC]:
                Z[edgeR, edgeC] = Gm[edgeR, edgeC]
        else:  # angle 135
            if Gm[edgeR, edgeC] >= Gm[edgeR-1, edgeC-1] and Gm[edgeR, edgeC] >= Gm[edgeR+1, edgeC+1]:
                Z[edgeR, edgeC] = Gm[edgeR, edgeC]
    return Z


def dilate(img, strel):
    assert img.ndim == 2, 'Image needs to be in 2d array'
    assert strel.ndim == 2, 'strel needs to be in 2d array'
    assert np.sum(strel) == 1, 'sum of strel needs to be equal to 1'

    if img.dtype == 'uint8':
        img /= 255.
    out = conv2d(img, strel)
    return (out > 0).astype('float32')


def erose(img, strel):
    assert img.ndim == 2, 'Image needs to be in 2d array'
    assert strel.ndim == 2, 'strel needs to be in 2d array'
    assert np.sum(strel) == 1, 'sum of strel needs to be equal to 1'

    if img.dtype == 'uint8':
        img /= 255.
    out = conv2d(img, strel)
    return (out == 1).astype('float32')


def histeq(img):  # Histogram equalization
    hist, bins = np.histogram(img.flatten(), 256, normed=True)
    cdf = hist.cumsum()
    cdf = 255*cdf/cdf[-1]
    imgeq = np.interp(img.flatten(), bins[:-1], cdf)
    return imgeq.reshape(img.shape)


def hough_circle_accumulator(edge_img, R_min=3, R_max=None, center_inside=True):
    assert edge_img.ndim == 2
    R, C = edge_img.shape
    if R_max is None:
        R_max = np.max((R, C))

    accumulator = np.zeros((R_max, R + 2*R_max, C + 2*R_max))
    thetas = np.linspace(-np.pi, np.pi, 360)[:-1]
    edges = np.argwhere(edge_img)

    for r in range(R_min, R_max):
        for edgeR, edgeC in edges:
            col = (r*np.cos(thetas)).astype('int32')
            row = (r*np.sin(thetas)).astype('int32')
            accumulator[r, edgeR+row+R_max, edgeC+col+R_max] += 1

    if center_inside:
        # center is inside the image
        return accumulator[:, R_max:R_max+R+1, R_max:R_max+C+1]
    else:
        return accumulator


def hough_line_accumulator(edge_img):
    assert edge_img.ndim == 2
    R, C = edge_img.shape
    D = int(np.ceil(np.sqrt(R**2 + C**2)))

    accumulator = np.zeros((2*D+1, 180))
    thetas = np.arange(180, dtype='int32')
    edges = np.argwhere(edge_img)

    for edgeR, edgeC in edges:
        p = edgeR*np.cos(thetas*np.pi/180) + edgeC*np.sin(thetas*np.pi/180)
        temp = (np.ceil(p + D + 1)).astype('int32')
        accumulator[temp, thetas] += 1
    return accumulator


def connected_component_labeling(bw):
    R, C = bw.shape
    out = np.zeros_like(bw) - 1.
    out = out.astype('int32')
    idx = np.argwhere(bw == 1)

    object_size = []
    label = 0
    for r, c in idx:
        if out[r, c] > -1:
            continue
        stack = []
        stack.append((r, c))
        object_size.append(0)
        while stack:
            r, c = stack.pop()
            if out[r, c] > -1:
                continue
            out[r, c] = label
            object_size[-1] += 1
            for i in range(max(r-1, 0), min(r+2, R)):
                for j in range(max(c-1, 0), min(c+2, C)):
                    if out[i, j] > -1 or bw[i, j] == 0:
                        continue
                    stack.append((i, j))
        label += 1
    return out, object_size


def imfill(bw):
    output_array = np.zeros_like(bw)
    output_array[1:-1, 1:-1] = 1.

    output_old_array = np.zeros_like(bw)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(bw, erose(output_array, np.ones((3, 3))/9.))
    return output_array


def hog_feature(img):
    from scipy.ndimage import uniform_filter
    img = rgb2gray(img, 'rgb') if img.ndim == 3 else np.at_least_2d(img)
    R, C = img.shape
    orientations = 9
    cx, cy = (8, 8)
    gx = np.zeros(img.shape)
    gy = np.zeros(img.shape)
    gx[:, :-1] = np.diff(img, n=1, axis=1)
    gy[:-1, :] = np.diff(img, n=1, axis=0)
    gmag = np.sqrt(gx**2 + gy**2)
    gorientation = np.arctan2(gy, (gx+1e-15)) * (180/np.pi) + 90
    nx = R//cx
    ny = C//cy
    orientation_hist = np.zeros((nx, ny, orientations))
    for i in range(orientations):
        temp = np.where(gorientation < 180 / orientations * (i+1), gorientation, 0)
        temp = np.where(gorientation >= 180 / orientations + i, temp, 0)
        cond2 = temp > 0
        mag = np.where(cond2, gmag, 0)
        orientation_hist[:,:,i] = uniform_filter(mag, size=(cx,cy))[cx//2::cx, cy//2::cy].T
    return orientation_hist.ravel()


def harris_corner_detector(img, threshold, kernel_size=3, p=0.5):
    if img.ndim == 3:
        img = rgb2gray(img)
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Iyy = dy**2
    Ixy = dy*dx
    R, C = img.shape
    K = np.ones((kernel_size, kernel_size), dtype='float32')
    offset = kernel_size//2
    Sxx = conv2d(Ixx, K)
    Syy = conv2d(Iyy, K)
    Sxy = conv2d(Ixy, K)
    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    respond = det - p*(trace**2)
    corners = np.argwhere(respond > threshold)
    return corners
