import math
import numpy as np
from itertools import chain
import pyopencl as cl

from mako.template import Template

from clarray import to_device
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

    __local ${Y.cl_buf.ocldtype} y_sum_post[${segment_size}][${dot_block_size}][${N_i}];
    __local int y_len, y_offset;
    __local int a_start[${dot_block_size}];
    __local int a_s0[${dot_block_size}];

% if n_dot_products is None :
    __local int n_dot_products;
% endif

    int local_idx = get_local_id(2)
        + get_local_size(2) * get_local_id(1)
        + get_local_size(2) * get_local_size(1) * get_local_id(0);
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
      y_offset = get_global_id(2);
    }
    y_sum_post[get_local_id(2)][get_local_id(1)][get_local_id(0)] = 0;


        if (local_idx < ${dot_block_size})
        {
            a_start[local_idx] = gstructure[1 * ${max_n_dots} + local_idx];
            a_s0[local_idx] = gstructure[2 * ${max_n_dots} + local_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

% if n_dot_products_ubound > dot_block_size :
    for (int ii = get_local_id(1);
             ii < ${n_dot_products_ubound};
             ii += ${dot_block_size})
    {
% else:
    {
% endif

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
% for ii in range(0, dot_block_size):
%    for jj in range(0, N_i):
%        if not (ii == jj == 0):
            y_sum_post[get_local_id(2)][0][0]
            += y_sum_post[get_local_id(2)][${ii}][${jj}];
%        endif
%    endfor
% endfor
    }

// write
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((local_idx < get_local_size(2)) && (y_offset + local_idx < y_len))
    {
        //XXX need to use global position!
    }
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
// write
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((local_idx < get_local_size(2)) && (y_offset + local_idx < y_len))
    {
        //XXX need to use global position!
    }
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

"""

kernel_template_2 = """
    __kernel void fn(
        const __global ${A.cl_buf.ocldtype} *A_data,
        __global ${Y.cl_buf.ocldtype} *Y_data)
{
    __local float buf[16][16];

    for (int ii = get_global_id(1); ii < 1600000; ii += get_global_size(1))
    {
        buf[get_local_id(0)][get_local_id(1)] = A_data[ii * 16 + get_local_id(0)];

        Y_data[ii] = ${float_alpha} * buf[get_local_id(0)][get_local_id(1)];
    }
}
"""

kernel_template_3 = """
    __kernel void fn(
        const __global ${A.cl_buf.ocldtype} *A_data,
        __global ${Y.cl_buf.ocldtype} *Y_data)
{
    __local float buf[${segment_size}][${N_i} * ${dot_block_size}];
    __local int y_offset;

    int local_idx = get_local_id(0)
        + get_local_size(0) * get_local_id(1)
        + get_local_size(0) * get_local_size(1) * get_local_id(2);
    int per_segment_idx = get_local_id(0)
        + get_local_size(0) * get_local_id(1);

    for (int ii = get_global_id(2); ii < 1600000 * 8; ii += get_global_size(2))
    {
        if (local_idx == 0)
        {
            y_offset = ii;
        }
        buf[get_local_id(2)][get_local_id(1) * ${N_i} + get_local_id(0)]
        = A_data[ii * 2 + get_local_id(0)];


        barrier(CLK_LOCAL_MEM_FENCE);

% for log2stride in range(n_reduce_steps):
        if (per_segment_idx + ${2 ** log2stride} < ${dot_block_size * N_i})
        {
        buf[get_local_id(2)][per_segment_idx]
            += buf[get_local_id(2)][per_segment_idx + ${2 ** log2stride}];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
% endfor
        if (local_idx < ${segment_size})
        {
            Y_data[y_offset + local_idx] = ${float_alpha} * buf[local_idx][0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
"""

kernel_template_4 = """
    __kernel void fn(
        const __global int *gstructure,
        const __global int *ii_lims,
        const __global int *dp_lims,
        const __global ${A.cl_buf.ocldtype} *A_data,
        __global ${Y.cl_buf.ocldtype} *Y_data)
{
    __local float buf[${segment_size}][${N_i} * ${dot_block_size}];
    __local int n_dot_products_bound, n_dot_products;
    __local y_len, y0_offset, y_offset, local_ii_lim;
%if shared_a_coords:
    __local int a_start[${dot_block_size}];
    __local int a_s0[${dot_block_size}];
% else:
    int a_start, a_s0;
% endif

    int local_idx = get_local_id(0)
        + get_local_size(0) * get_local_id(1)
        + get_local_size(0) * get_local_size(1) * get_local_id(2);
    int per_segment_idx = get_local_id(0)
        + get_local_size(0) * get_local_id(1);
    int bb0 = get_global_id(2) / ${segments_per_y * segment_size};
    int ii0 = get_global_id(2) % ${segments_per_y * segment_size};

    //printf("bb, local_idx: %i %i\\n", bb0, local_idx);

    for (int bb = bb0; bb < ${n_items}; bb += ${item_skip})
    {
        if (local_idx == 0)
        {
            n_dot_products_bound = dp_lims[bb];
            n_dot_products = gstructure[
                bb * ${structure_vars_stride} + 4 * ${max_n_dots} + 2];
            y_len = gstructure[
                bb * ${structure_vars_stride} + 4 * ${max_n_dots} + 3];
            local_ii_lim = ii_lims[bb];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ii = ii0;
                 ii < local_ii_lim;
                 ii += ${segments_per_y * segment_size})
        {
            if (local_idx == 0)
            {
                y0_offset = gstructure[bb * ${structure_vars_stride}
                                           + 4 * ${max_n_dots} + 1];
                y_offset = ii; 
            }
            buf[get_local_id(2)][get_local_id(1) * ${N_i} + get_local_id(0)] = 0;
            //printf("n_dots_bound: %i\\n", n_dot_products_bound);
            for (int jj = get_local_id(1); jj < n_dot_products_bound; jj += get_local_size(1))
            {
% if shared_a_coords:
                if ((get_local_id(0) == 0)  && (get_local_id(2) == 0))
                {
                    a_start[get_local_id(1)] = gstructure[
                        bb * ${structure_vars_stride} + 1 * ${max_n_dots} + jj];
                    a_s0[get_local_id(1)] = gstructure[
                        bb * ${structure_vars_stride} + 2 * ${max_n_dots} + jj];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if ((jj < n_dot_products) && (ii < y_len))
                {
                    buf[get_local_id(2)][get_local_id(1) * ${N_i} + get_local_id(0)]
                        += A_data[a_start[get_local_id(1)]
                                  + ii * a_s0[get_local_id(1)]
                                  + get_local_id(0)];
                }
% else:
                a_start = gstructure[bb * ${structure_vars_stride} + 1 * ${max_n_dots} + jj];
                a_s0 = gstructure[bb * ${structure_vars_stride} + 2 * ${max_n_dots} + jj];
                if ((jj < n_dot_products) && (ii < y_len))
                {
                    buf[get_local_id(2)][get_local_id(1) * ${N_i} + get_local_id(0)]
                        += A_data[a_start + ii * a_s0 + get_local_id(0)];
                }
% endif
                barrier(CLK_LOCAL_MEM_FENCE);
            }

% for log2stride in range(n_reduce_steps):
            if (per_segment_idx + ${2 ** log2stride} < ${dot_block_size * N_i})
            {
                buf[get_local_id(2)][per_segment_idx]
                    += buf[get_local_id(2)][per_segment_idx + ${2 ** log2stride}];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
% endfor

            if (   (y_offset + local_idx < y_len)
                && (local_idx < ${segment_size}))
            {
                //printf("y_offset: %i %i %i\\n", y_offset, bb0, local_idx);
                Y_data[y0_offset + y_offset + local_idx]
                  = ${float_alpha} * buf[local_idx][0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
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

    # XXX THESE DEFAULTS CAN'T BE SET REASONABLY
    #     The best parameters depend strongly on the problem configuration.
    #     E.g.
    #        * register use vs. local variables
    #        * max segment size should be big, but not at expense of occupancy
    MAX_SEGMENT_SIZE = 8 # HYPER
    MAX_ITEM_SKIP = 8 # HYPER
    MAX_SEGMENTS_IN_FLIGHT = 500

    segment_size = min(
        max_y_len,
        MAX_SEGMENT_SIZE)
    dot_block_size = min(
        max_n_dots,
        int(p.queue.device.max_work_group_size
            / segment_size
            / N_i),
        )

    item_skip = min(MAX_ITEM_SKIP, len(items))

    n_dot_blocks, n_dot_products_ubound = padsize(max_n_dots, dot_block_size)

    n_segments, ylen_ubound = padsize(max_y_len, segment_size)
    segments_per_y = min(n_segments, MAX_SEGMENTS_IN_FLIGHT / item_skip)
    gsize = (N_i, dot_block_size, segment_size * segments_per_y * item_skip)
    lsize = (N_i, dot_block_size, segment_size)

    n_reduce_steps = int(math.ceil(np.log2(dot_block_size * N_i)))

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
        'n_reduce_steps': n_reduce_steps,
        'segments_per_y': segments_per_y,
        'item_skip': item_skip,
        'shared_a_coords': False,
        })
    if 1:
        for k, v in textconf.items():
            print k, v
    textconf.update(p.__dict__)

    if 0:
        gsize = (16, 1600)
        lsize = (16, 16)

        text = Template(kernel_template_2, output_encoding='ascii').render(
            **textconf)
    elif 0:
        text = Template(kernel_template_3, output_encoding='ascii').render(
            **textconf)
    else:
        text = Template(kernel_template_4, output_encoding='ascii').render(
            **textconf)

    fn = cl.Program(p.queue.context, text).build().fn

    ii_lims = [
        math.ceil(p.geometry[ii]['y_len'] / float(segment_size)) * segment_size
        for ii in items]
    cl_ii_lims = to_device(p.queue, np.asarray(ii_lims, dtype='int32'))
    dp_lims = [
        math.ceil(len(p.geometry[ii]['dots']) / float(dot_block_size)) * dot_block_size
        for ii in items]
    cl_dp_lims = to_device(p.queue, np.asarray(dp_lims, dtype='int32'))

    full_args = [
        cl_gstructure,
        cl_ii_lims,
        cl_dp_lims,
        p.A.cl_buf,
        #p.X.cl_buf,
        ]
    #if p.cl_beta is not None:
        #full_args += [p.cl_beta]
    full_args += [
                 #p.Y_in.cl_buf,
                 p.Y.cl_buf,
                ]
    print gsize, lsize

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

