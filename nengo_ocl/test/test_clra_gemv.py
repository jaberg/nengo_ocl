
import numpy as np

from nengo_ocl.tricky_imports import unittest
from  nengo_ocl import raggedarray as ra
RA = ra.RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.clra_gemv import plan_ragged_gather_gemv
from nengo_ocl.clra_gemv import plan_many_dots
from nengo_ocl.clra_gemv import plan_reduce
from nengo_ocl.clra_gemv import plan_ref

import pyopencl as cl
import logging

ctx = cl.create_some_context()
logger = logging.getLogger(__name__)

def allclose(raA, raB):
    assert len(raA) == len(raB)
    for i in xrange(len(raA)):
         if not np.allclose(raA[i], raB[i]):
             return False
    return True

class TestStuff(unittest.TestCase):

    def test_basic(self):
        # -- prepare initial conditions on host
        A = RA([ [[0.1, .2], [.3, .4]], [[.5, .6]]])
        X = RA([ [3, 5] ])
        Y = RA([[0.0], [2, 3],])
        A_js = RA([[1], [0]])
        X_js = RA([[0], [0]])
        alpha = 0.5
        beta = 0.1

        # -- prepare initial conditions on device
        queue = cl.CommandQueue(ctx)
        clA = CLRA(queue, A)
        clX = CLRA(queue, X)
        clY = CLRA(queue, Y)
        clA_js = CLRA(queue, A_js)
        clX_js = CLRA(queue, X_js)
        assert allclose(A, clA)
        assert allclose(X, clX)
        assert allclose(Y, clY)
        assert allclose(A_js, clA_js)
        assert allclose(X_js, clX_js)

        # -- run cl computation
        plan = plan_ragged_gather_gemv(
            queue, alpha, clA, clA_js, clX, clX_js, beta, clY)

        plan()

        # -- ensure they match
        for i in xrange(len(A_js)):
            aj, xj = int(A_js[i]), int(X_js[i])
            ref = alpha*np.dot(A[aj], X[xj]) + beta*Y[i]
            sim = clY[i]
            assert np.allclose(ref, sim)

    def _test_random(self, k=4, p=1, m=10, n=10):
        """
        Parameters
        ----------
        k : number of operations (length of A_js)
        p : number of dots per operation (width of A_js)
        m : output dimensions
        n : input dimensions
        """

        rng = np.random.RandomState(3294)

        aa = [rng.normal(size=(m, n)) for i in xrange(k)]
        xx = [rng.normal(size=n) for i in xrange(k)]
        yy = [rng.normal(size=m) for i in xrange(k)]
        ajs = [rng.randint(k, size=p) for i in xrange(k)]
        xjs = [rng.randint(k, size=p) for i in xrange(k)]

        A = RA(aa)
        X = RA(xx)
        Y = RA(yy)
        A_js = RA(ajs)
        X_js = RA(xjs)
        alpha = 0.5
        beta = 0.1

        # -- prepare initial conditions on device
        queue = cl.CommandQueue(ctx)
        clA = CLRA(queue, A)
        clX = CLRA(queue, X)
        clY = CLRA(queue, Y)
        clA_js = CLRA(queue, A_js)
        clX_js = CLRA(queue, X_js)
        assert allclose(A, clA)
        assert allclose(X, clX)
        assert allclose(Y, clY)
        assert allclose(A_js, clA_js)
        assert allclose(X_js, clX_js)

        # -- run cl computation
        prog = plan_ragged_gather_gemv(
            queue, alpha, clA, clA_js, clX, clX_js, beta, clY)

        print '-' * 5 + ' Plans ' + '-' * 45
        for plan in prog.plans:
            print plan
        prog()

        # -- ensure they match
        for i in xrange(k):
            ref = beta*Y[i]
            for aj, xj in zip(A_js[i], X_js[i]):
                ref += alpha*np.dot(A[aj], X[xj])
            sim = clY[i]
            assert np.allclose(ref, sim, atol=1e-3, rtol=1e-3)

    def test_random_small(self):
        self._test_random(k=4, m=10, n=10)

    def test_random_large(self):
        self._test_random(k=10, m=550, n=550)

    def test_many_dots_small(self):
        self._test_random(k=4, p=4, m=10, n=10)

    def test_many_dots_large(self):
        # self._test_random(k=4, p=4, m=550, n=550)
        self._test_random(k=4, p=4, m=2000, n=1000)


def check_from_shapes(
    planner,
    alpha, beta, gamma,
    A_shapes, X_shapes,
    A_js,
    X_js,
    ):
    rng = np.random.RandomState(1234)
    A = RA([0.1 + rng.rand(*shp) for shp in A_shapes])
    X = RA([0.1 + rng.rand(*shp) for shp in X_shapes])
    Y = RA([0.1 + rng.rand(
        A_shapes[A_js[ii][0]][0],
        X_shapes[X_js[ii][0]][1])
        for ii in range(len(A_js))])
    A_js = RA(A_js)
    X_js = RA(X_js)
    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    clA_js = CLRA(queue, A_js)
    clX_js = CLRA(queue, X_js)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)
    assert allclose(A_js, clA_js)
    assert allclose(X_js, clX_js)

    # -- run cl computation
    plan = planner(
        queue, alpha, clA, clA_js, clX, clX_js, beta, clY,
        gamma=gamma)

    plan()

    # -- ensure they match
    for i in xrange(len(A_js)):
        #print 'gamma', gamma
        #print 'Y[i] * beta + gamma', Y[i] * beta + gamma
        #print A[0]
        #print X[0]
        #print 'AX', sum(
            #[np.dot(A[aj], X[xj])
             #for aj, xj in zip(A_js[i], X_js[i])])
        ref = gamma + beta * Y[i] + alpha * sum(
            [np.dot(A[aj], X[xj])
             for aj, xj in zip(A_js[i], X_js[i])])
        sim = clY[i]
        if not np.allclose(ref, sim, atol=1e-3, rtol=1e-3):
            print 'A_shapes',  A_shapes
            print 'X_shapes', X_shapes
            if len(ref) > 20:
                print 'ref', ref[:10], '...', ref[-10:]
                print 'sim', sim[:10], '...', sim[-10:]
            else:
                print 'ref', ref
                print 'sim', sim
            assert 0

class ShapeCheckMixin(object):
    def test_basic(self):
        self.check_from_shapes(
            0.5, 0.6, 0.7,
            A_shapes = [(1, 1)],
            X_shapes = [(1, 1)],
            A_js = [[0]],
            X_js = [[0]])

    def test_one_short_segment(self):
        self.check_from_shapes(
            0.5, 0.6, 0.7,
            A_shapes = [(10, 1)],
            X_shapes = [(1, 1)],
            A_js = [[0]],
            X_js = [[0]])

    def test_one_long_segment(self):
        self.check_from_shapes(
            0.5, 0.6, 0.7,
            A_shapes = [(2001, 1)],
            X_shapes = [(1, 1)],
            A_js = [[0]],
            X_js = [[0]])

    def test_one_short_segment_many_dots(self):
        for ND in 2, 20, 100:
            self.check_from_shapes(
                0.5, 0.6, 0.7,
                A_shapes = [(10, 1 + ii % 2) for ii in range(ND)],
                X_shapes = [(1 + ii % 2, 1) for ii in range(ND)],
                A_js = [range(ND)],
                X_js = [range(ND)])

    def test_one_short_segment_many_longer_dots(self):
        for ND in 2, 20, 100:
            self.check_from_shapes(
                0.5, 0.6, 0.7,
                A_shapes = [(2000, ii + 1) for ii in range(ND)],
                X_shapes = [(ii + 1, 1) for ii in range(ND)],
                A_js = [range(ND)],
                X_js = [range(ND)])

class TestManyDots(unittest.TestCase, ShapeCheckMixin):

    def check_from_shapes(self, *args, **kwargs):
        return check_from_shapes(plan_many_dots, *args, **kwargs)

class TestReduce(unittest.TestCase, ShapeCheckMixin):

    def check_from_shapes(self, *args, **kwargs):
        return check_from_shapes(plan_reduce, *args, **kwargs)

class TestRef(unittest.TestCase, ShapeCheckMixin):

    def check_from_shapes(self, *args, **kwargs):
        return check_from_shapes(plan_ref, *args, **kwargs)

if __name__ == '__main__':

   unittest.main()

