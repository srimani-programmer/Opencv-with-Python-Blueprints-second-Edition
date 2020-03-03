import numpy as np
import numba

def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_tl, a_br = a[:4].reshape((2, 2))
    b_tl, b_br = b[:4].reshape((2, 2))
    int_tl = np.maximum(a_tl, b_tl)
    int_br = np.minimum(a_br, b_br)
    int_area = np.product(np.maximum(0., int_br - int_tl))
    a_area = np.product(a_br - a_tl)
    b_area = np.product(b_br - b_tl)

    return int_area / (a_area + b_area - int_area)


def calc_iou_matrix(detections,trackers):
    iou_matrix = np.zeros((len(detections), len(trackers)),dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    return iou_matrix

A = np.random.rand(100,4)
B = np.random.rand(100,4)

print(calc_iou_matrix(A,B))