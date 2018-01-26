import cupy
import numpy

import cupy_perf


class Perf(cupy_perf.PerfCases):
    def setUp(self, n):
        self.n = n
        self.a = cupy.empty(self.n, dtype=numpy.complex64)


class Perf_16(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_16, self).setUp(1 << 16)

    @cupy_perf.attr(n=100)
    def perf_fft(self):
        cupy.fft.fft(self.a)


class Perf_20(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_20, self).setUp(1 << 20)

    @cupy_perf.attr(n=100)
    def perf_fft(self):
        cupy.fft.fft(self.a)


class Perf_24(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24, self).setUp(1 << 24)

    @cupy_perf.attr(n=100)
    def perf_fft(self):
        cupy.fft.fft(self.a)


class Perf_Plan_16(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_Plan_16, self).setUp(1 << 16)
        self.plan = cupy.cuda.cufft.Plan1d(self.n, cupy.cuda.cufft.CUFFT_C2C, 1)
        self.out = cupy.arange(self.n).astype(numpy.complex64)

    @cupy_perf.attr(n=100)
    def perf_fft(self):
        self.plan.fft(self.a, self.out, cupy.cuda.cufft.CUFFT_FORWARD)


class Perf_Plan_20(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_Plan_20, self).setUp(1 << 20)
        self.plan = cupy.cuda.cufft.Plan1d(self.n, cupy.cuda.cufft.CUFFT_C2C, 1)
        self.out = cupy.arange(self.n).astype(numpy.complex64)

    @cupy_perf.attr(n=100)
    def perf_fft(self):
        self.plan.fft(self.a, self.out, cupy.cuda.cufft.CUFFT_FORWARD)


class Perf_Plan_24(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_Plan_24, self).setUp(1 << 24)
        self.plan = cupy.cuda.cufft.Plan1d(self.n, cupy.cuda.cufft.CUFFT_C2C, 1)
        self.out = cupy.arange(self.n).astype(numpy.complex64)

    @cupy_perf.attr(n=100)
    def perf_fft(self):
        self.plan.fft(self.a, self.out, cupy.cuda.cufft.CUFFT_FORWARD)


cupy_perf.run(__name__)
