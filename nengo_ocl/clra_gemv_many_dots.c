
__kernel void fn(
    const __global int *gstructure,
    const __global ${A.cl_buf.ocldtype} *A_data,
    const __global ${X.cl_buf.ocldtype} *X_data,
    % if cl_beta is not None:
    const __global ${cl_beta.ocldtype} * betas,
    % endif
    const __global ${Y_in.cl_buf.ocldtype} *Y_in_data,
    __global ${Y.cl_buf.ocldtype} *Y_data)
{
  __local int lstructure[4 * ${dot_block_size}];
  __local ${Y.cl_buf.ocldtype} y_sum_pre[${segment_size}];
  __local ${Y.cl_buf.ocldtype} \
      y_sum_post[${dot_block_size}][${segment_size}];

  const int local_idx = get_local_id(0) \
      + get_local_id(1) * get_local_size(0);

  int segment_idx = get_local_id(0);
  int dot_block_idx = get_local_id(1);

  if (get_global_id(0) < ${y_len})
    {

      if (dot_block_idx == 0)
        {
% if float_beta is not None and float_beta != 0 :
          y_sum_pre[segment_idx]
          = ${float_beta} * Y_in_data[${y_in_starts} + get_global_id(0)];
% elif cl_beta is not None:
          y_sum_pre[segment_idx]
          = betas[${bb}] * Y_in_data[${y_in_starts} + get_global_id(0)];
% else :
          y_sum_pre[segment_idx] = 0;
% endif

% if float_gamma is not None:
  % if float_gamma != 0:
          y_sum_pre[segment_idx] += ${float_gamma};
  % endif
% endif
        }

    y_sum_post[dot_block_idx][segment_idx] = 0;
    for (int ii = dot_block_idx;
             ii < ${n_dot_products};
             ii += ${dot_block_size})
    {
% if segment_size == 1:
        lstructure[get_local_id(1)][0] = gstructure[0 * ${max_n_dots} + ii];
        lstructure[get_local_id(1)][1] = gstructure[1 * ${max_n_dots} + ii];
        lstructure[get_local_id(1)][2] = gstructure[2 * ${max_n_dots} + ii];
        lstructure[get_local_id(1)][3] = gstructure[3 * ${max_n_dots} + ii];
% elif segment_size == 2:
        lstructure[get_local_id(1)][get_local_id(0)]
          = gstructure[get_local_id(0) * ${max_n_dots} + ii];
        lstructure[get_local_id(1)][get_local_id(0) + 2]
          = gstructure[(get_local_id(0) + 2) * ${max_n_dots} + ii];
% elif segment_size == 4:
        lstructure[get_local_id(1)][get_local_id(0)]
          = gstructure[get_local_id(0) * ${max_n_dots} + ii];
% else:
        for (int jj = get_local_id(0); jj < 4; jj += get_local_size(0))
          {
            lstructure[get_local_id(1)][jj] = gstructure[ii * 4 + jj];
          }
% endif
        for (int nn = 0; nn < ${N_i}; nn += 1)
          {
            y_sum_post[dot_block_idx][segment_idx]
              += A_data[${a_starts} + get_global_id(0) * ${a_s0} + nn]
                 * X_data[${x_starts} + nn];
          }
    }
}
barrier(CLK_LOCAL_MEM_FENCE);
//printf("AX=%f\\n", y_sum_post[dot_block_idx][segment_idx]);
if ((get_global_id(0) < ${y_len}) && (dot_block_idx == 0))
{
    for (int ii = 1; ii < ${dot_block_size}; ++ii)
    {
        y_sum_post[0][segment_idx] += y_sum_post[ii][segment_idx];
    }
    Y_data[${y_offset} + get_global_id(0)]
        = y_sum_pre[segment_idx]
          + ${float_alpha} * y_sum_post[0][segment_idx];
//printf("Yout=%f\\n", Y_data[${y_offset} + get_global_id(0)]);
}
}
