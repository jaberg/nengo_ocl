import math
from itertools import chain
import pyopencl as cl

from mako.template import Template

from plan import Plan
from clra_gemv import (
    gemv_prog,
    bw_from_geometry,
    flops_from_geometry)

def padsize(a, q):
    n = int(math.ceil(float(a) / q))
    return n, n * q


"""
    __local ${X.cl_buf.ocldtype} lX[${dot_block_size}][${N_i}];
    __local ${Y.cl_buf.ocldtype} y_sum_pre[${segment_size}];
    __local int geometry_anchor;
    geometry_anchor = 0;
            geometry_anchor + 
            geometry_anchor + 
        geometry_anchor += ${dot_block_size};

    gstructure += get_global_id(2) * ${structure_vars_stride};

        if (get_local_id(2) == 0)
        {
            lX[get_local_id(1)][get_local_id(0)] = X_data[gstructure[ii] + get_local_id(0)];
        }

                 * lX[get_local_id(0)];
"""

kernel_template = """
    __kernel void fn(
        const __global int *gstructure,
        const __global ${A.cl_buf.ocldtype} *A_data,
        __global ${Y.cl_buf.ocldtype} *Y_data)
{
    __local ${Y.cl_buf.ocldtype} y_sum_post[${segment_size}][${dot_block_size}][${N_i}];
    __local int y_len;
    __local int a_start[${dot_block_size}];
    __local int a_s0[${dot_block_size}];

% if n_dot_products is None :
    __local int n_dot_products;
% endif

    int local_idx = get_local_id(0)
        + get_local_size(0) * get_local_id(1)
        + get_local_size(0) * get_local_size(1) * get_local_id(2);
    // int bb = 0;  // XXX: handle multiple output vectors

    // XXX: if possible, use one wavefront to read this
    //                       another wavefront to read a_start
    //                       a third to read a_s0
    //                       a fourth to read lX

    if (local_idx == 0)
    {
% if n_dot_products is None :
      n_dot_products = gstructure[0 * ${structure_vars_stride} + 4 * ${max_n_dots} + 2]
% endif
      y_len = gstructure[0 * ${structure_vars_stride} + 4 * ${max_n_dots} + 3];
    }
    y_sum_post[get_local_id(2)][get_local_id(1)][get_local_id(0)] = 0;

% if n_dot_products_ubound > dot_block_size :
    for (int ii = get_local_id(1);
             ii < ${n_dot_products_ubound};
             ii += ${dot_block_size})
    {
% else:
    {
% endif

        if (local_idx < ${dot_block_size})
        {
            a_start[local_idx] = gstructure[1 * ${max_n_dots} + local_idx];
            a_s0[local_idx] = gstructure[2 * ${max_n_dots} + local_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (1
            && (get_global_id(0) < ${N_i})
% if n_dot_products is None:
            && (get_global_id(1) < n_dot_products)
% else:
            && (get_global_id(1) < ${n_dot_products})
% endif
            && (get_global_id(2) < y_len))
        {
            y_sum_post[get_local_id(2)][get_local_id(1)][get_local_id(0)]
                += A_data[a_start[get_local_id(1)]
                          + get_global_id(2) * a_s0[get_local_id(1)]
                          + get_local_id(0)];
        }
% if n_dot_products_ubound > dot_block_size :
        barrier(CLK_LOCAL_MEM_FENCE);
    }
% else:
    }
% endif


// short reduction XXX: use an log-time reduce
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((get_global_id(2) < y_len) && (get_local_id(0) == 0) && (get_local_id(1) == 0))
    {
% for ii in range(1, dot_block_size):
%    for jj in range(1, N_i):
         y_sum_post[get_local_id(2)][0][0]
         += y_sum_post[get_local_id(2)][${ii}][${jj}];
%    endfor
% endfor
    }

// write
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_idx < y_len)
    {
        //XXX need to use global position!
        Y_data[local_idx]
            = 0.0 // XXX y_sum_pre[local_idx]
            + ${float_alpha} * y_sum_post[local_idx][0][0];
    }
}
"""

def plan0_impl(p, items):

    p.print_geometry_summary(items)

    A_js_shape0s = p.A_js.shape0s
    cl_gstructure, textconf = p.cl_geometry_and_textconf(items)

    max_n_dots = max(A_js_shape0s)

    max_y_len = max(p.geometry[ii]['y_len'] for ii in items)
    N_is = list(set(chain(*[
        [d['a_shape1'] for d in p.geometry[ii]['dots']]
        for ii in items])))
    try:
        N_i, = N_is
    except IndexError:
        raise NotImplementedError('only single N_i supported')
    try:
        n_dot_products, = list(set([len(p.geometry[ii]['dots']) for ii in
            items]))
    except IndexError:
        n_dot_products = None

    MAX_SEGMENT_SIZE = 8 # tricky to tune?

    segment_size = min(
        max_y_len,
        MAX_SEGMENT_SIZE)
    dot_block_size = min(
        max_n_dots,
        int(p.queue.device.max_work_group_size / segment_size),
        )

    n_dot_blocks, n_dot_products_ubound = padsize(max_n_dots, dot_block_size)

    n_segments, ylen_ubound = padsize(max_y_len, segment_size)
    gsize = (N_i, dot_block_size, ylen_ubound)
    lsize = (N_i, dot_block_size, segment_size)
    print gsize, lsize

    textconf.update({
        'gsize': gsize,
        'lsize': lsize,
        'n_items': len(items),
        'segment_size': segment_size,
        'dot_block_size': dot_block_size,
        'N_i': N_i,
        'max_y_len': max_y_len,
        'n_dot_products_ubound': n_dot_products_ubound,
        'n_dot_products': n_dot_products,
        })
    if 1:
        for k, v in textconf.items():
            print k, v
    textconf.update(p.__dict__)

    text = Template(kernel_template, output_encoding='ascii').render(
        **textconf)

    fn = cl.Program(p.queue.context, text).build().fn

    full_args = [
                 cl_gstructure,
                 p.A.cl_buf,
                 #p.X.cl_buf,
                 ]
    #if p.cl_beta is not None:
        #full_args += [p.cl_beta]
    full_args += [
                 #p.Y_in.cl_buf,
                 p.Y.cl_buf,
                ]

    fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(p.queue, fn, gsize, lsize,
        name='clra_gemv.plan_0',
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items),
        )
    rval.full_args = full_args  # prevent GC the args
    return rval


class plan_gemv0(gemv_prog):
    def choose_plans(self):
        return [plan0_impl(self, range(len(self.Y)))]

