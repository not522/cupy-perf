import cupy
import cupyx.scipy.ndimage
import cv2
import numpy
import scipy.ndimage

import cupy_perf


class Perf(cupy_perf.PerfCases):
    def setUp(self, n, sz):
        shape = (sz, sz, n)
        self.a = cupy.empty(shape, numpy.float32)

    def affine(self):
        cupyx.scipy.ndimage.affine_transform(self.a, cupy.array([[1., 0., 0., -2.5], [0., 0.5, 0., 0.], [0., 0., 1., 0.]]), order=1)

    def rotate(self):
        cupyx.scipy.ndimage.rotate(self.a, 1, order=1, reshape=False)

    def zoomin(self):
        cupyx.scipy.ndimage.zoom(self.a, [2, 2, 1], order=1)

    def zoomout(self):
        cupyx.scipy.ndimage.zoom(self.a, [0.5, 0.5, 1], order=1)


class Perf_cpu(cupy_perf.PerfCases):
    def setUp(self, n, sz):
        shape = (sz, sz, n)
        self.a = numpy.empty(shape, numpy.float32)

    def affine(self):
        scipy.ndimage.affine_transform(self.a, [[1., 0., 0., -2.5], [0., 0.5, 0., 0.], [0., 0., 1., 0.]], order=1)

    def rotate(self):
        scipy.ndimage.rotate(self.a, 1, order=1, reshape=False)

    def zoomin(self):
        scipy.ndimage.zoom(self.a, [2, 2, 1], order=1)

    def zoomout(self):
        scipy.ndimage.zoom(self.a, [0.5, 0.5, 1], order=1)


class Perf_opencv(cupy_perf.PerfCases):
    def setUp(self, n, sz):
        shape = (sz, sz, n)
        self.a = numpy.empty(shape, numpy.float32)
        # cv2.ocl.setUseOpenCL(True)

    def affine(self):
        M = numpy.array([[2., 0., 0.], [0., 1., 5.]])
        cv2.warpAffine(self.a, M, self.a.shape[:2])

    def rotate(self):
        M = cv2.getRotationMatrix2D((self.a.shape[1] / 2.0 - 0.5, self.a.shape[0] / 2.0 - 0.5), 1, 1)
        cv2.warpAffine(self.a, M, self.a.shape[:2])

    def zoomin(self):
        shape = list(self.a.shape)[:2]
        shape[0] *= 2
        shape[1] *= 2
        cv2.resize(self.a, tuple(shape))

    def zoomout(self):
        shape = list(self.a.shape)[:2]
        shape[0] //= 2
        shape[1] //= 2
        cv2.resize(self.a, tuple(shape))


class Perf_100_100(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_100_100, self).setUp(100, 100)

    @cupy_perf.attr(n=100)
    def perf_affine(self):
        self.affine()

    @cupy_perf.attr(n=100)
    def perf_rotate(self):
        self.rotate()

    @cupy_perf.attr(n=100)
    def perf_zoomin(self):
        self.zoomin()

    @cupy_perf.attr(n=100)
    def perf_zoomout(self):
        self.zoomout()

'''
class Perf_100_100_cpu(Perf_cpu):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_100_100_cpu, self).setUp(100, 100)

    @cupy_perf.attr(n=100)
    def perf_affine(self):
        self.affine()

    @cupy_perf.attr(n=100)
    def perf_rotate(self):
        self.rotate()

    @cupy_perf.attr(n=100)
    def perf_zoomin(self):
        self.zoomin()

    @cupy_perf.attr(n=100)
    def perf_zoomout(self):
        self.zoomout()
'''

class Perf_100_100_opencv(Perf_opencv):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_100_100_opencv, self).setUp(100, 100)

    @cupy_perf.attr(n=100)
    def perf_affine(self):
        self.affine()

    @cupy_perf.attr(n=100)
    def perf_rotate(self):
        self.rotate()

    @cupy_perf.attr(n=100)
    def perf_zoomin(self):
        self.zoomin()

    @cupy_perf.attr(n=100)
    def perf_zoomout(self):
        self.zoomout()


cupy_perf.run(__name__)
