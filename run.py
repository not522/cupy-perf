import cupy
import numpy

import cupy_perf


class Perf(cupy_perf.PerfCases):
    def setUp(self, n, k):
        self.n = n
        self.k = k
        self.a = cupy.arange(self.n).astype('f')
        self.orig = self.a.copy()
        cupy.random.shuffle(self.a)

    def tearDown(self):
        self.a = cupy.partition(self.a, self.k)
        assert numpy.array_equal(cupy.asnumpy(cupy.sort(self.orig)), cupy.asnumpy(cupy.sort(self.a)))
        assert cupy.max(self.a[:self.k]) <= self.a[self.k]
        assert cupy.min(self.a[self.k:]) >= self.a[self.k]


class Perf_12_20(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_12_20, self).setUp(1 << 12, 20)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_16_20(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_16_20, self).setUp(1 << 16, 20)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_20_20(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_20_20, self).setUp(1 << 20, 20)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_24_20(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24_20, self).setUp(1 << 24, 20)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_24_10(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24_10, self).setUp(1 << 24, 10)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_24_100(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24_100, self).setUp(1 << 24, 100)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_24_1000(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24_1000, self).setUp(1 << 24, 1000)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


cupy_perf.run(__name__)
