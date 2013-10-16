import sys

import numpy as np
import pyopencl as cl

from nengo_ocl.tricky_imports import unittest
from  nengo_ocl import raggedarray as ra
RA = ra.RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.test.test_clra_gemv import (
        ShapeCheckMixin,
        check_from_shapes,
        ctx,
        allclose)
from nengo_ocl.clra_gemv_0 import plan_gemv0


class TestGemv0(unittest.TestCase, ShapeCheckMixin):

    def check_from_shapes(self, *args, **kwargs):
        return check_from_shapes(plan_gemv0 *args, **kwargs)


    def test_read_a(self):
        alpha = 1.0
        beta = 0.0
        gamma = 0.0

        A_shapes = [(16, 1)]
        X_shapes = [(1, 1)]
        A_js = [[0]]
        X_js = [[0]]

        rng = np.random.RandomState(1234)
        A = RA([0.1 + rng.rand(*shp) for shp in A_shapes])
        X = RA([np.ones(shp) for shp in X_shapes])  #DEBUG
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
        plan = plan_gemv0(
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


if __name__ == '__main__':
   sys.exit(unittest.main())
