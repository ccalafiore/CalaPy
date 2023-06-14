import cv2


def read_image(directory, order='RGB'):
    array_bgr = cv2.imread(directory, cv2.IMREAD_COLOR)
    order_l = order.upper()
    if order_l == 'RGB':
        array_rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
        # array_rgb = array_bgr[:, :, slice(-1, -4, -1)]
        return array_rgb
    elif order_l == 'BGR':
        return array_bgr
    else:
        raise ValueError('order')
