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


class Perf_16_31(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_16_31, self).setUp(1 << 16, 31)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_16_63(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_16_63, self).setUp(1 << 16, 63)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_16_127(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_16_127, self).setUp(1 << 16, 127)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_20_31(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_20_31, self).setUp(1 << 20, 31)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_20_63(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_20_63, self).setUp(1 << 20, 63)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_20_127(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_20_127, self).setUp(1 << 20, 127)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_24_31(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24_31, self).setUp(1 << 24, 31)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_24_63(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24_63, self).setUp(1 << 24, 63)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


class Perf_24_127(Perf):
    def setUp(self):
        print(self.__class__.__name__)
        super(Perf_24_127, self).setUp(1 << 24, 127)

    @cupy_perf.attr(n=100)
    def perf_partition(self):
        cupy.partition(self.a, self.k)


cupy_perf.run(__name__)
