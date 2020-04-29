import triton
import torch
from .matmul import _sparse_matmul
import math

src = '''
  __global__ void NAME (TYPE* A __readonly  __noalias __aligned(16),
                        TYPE* B __readonly  __noalias __aligned(16),
                        TYPE* C __noalias __aligned(16),
                        // shapes
                        int H, int W, int R, int S, int CC,
                        int N, int P, int Q, int K __multipleof(BLOCK),
                        int pad_h, int pad_w,
                        int stride_h, int stride_w,
                        // a strides
                        int stride_na __multipleof(BLOCK),
                        int stride_ha __multipleof(BLOCK),
                        int stride_wa __multipleof(BLOCK),
                        // c strides
                        int stride_nc __multipleof(BLOCK),
                        int stride_hc __multipleof(BLOCK),
                        int stride_wc __multipleof(BLOCK),
                        // lut and locks
                        int* lut, int* locks, int nlocks) {
     /* ---------------- */
    /*    Prologue      */
    /* ---------------- */
    // program ids
    int pid0 = get_program_id(0);
    int pid1 = get_program_id(1);
#ifdef DW
    int* header = lut + pid0 * 4;
    int  off_ck = *(header + 0);
    int  off_cc = *(header + 1);
    int  off_cr = *(header + 2);
    int  off_cs = *(header + 3);
    int L = N*P*Q;
    int lockid = 0;
    int maxid = TZ;
    int* p_delta = lut + get_num_programs(0)*4;
    int* pa_delta[TL] = p_delta + 0 ... TL;
    int* pb_delta[TL] = p_delta + N*P*Q + TL + 0 ... TL;
    int  a_delta[TL]  = *pa_delta;
    int  b_delta[TL]  = *pb_delta;
    int  ra_c[TM] = off_cc * TM + 0 ... TM;
    int  rb_k[TN] = off_ck * TN + 0 ... TN;
    TYPE* pa[TM, TL] = A + ra_c[:, newaxis] * 1
                         + off_cr * stride_ha
                         + off_cs * stride_wa
                         + a_delta[newaxis, :];
    TYPE* pb[TL, TN] = B + rb_k[newaxis, :] * 1
                         + b_delta[:, newaxis];
    int  ra_nhw[TL] = 0 ... TL;
    int  ra_nh[TL]  = ra_nhw / Q;
    int  ra_w [TL]  = (ra_nhw % Q)*stride_w;
    int  ra_h [TL]  = (ra_nh % P)*stride_h;
    int  h_lo = 0 + pad_h - off_cr;
    int  h_hi = H + pad_h - off_cr;
    int  w_lo = 0 + pad_w - off_cs;
    int  w_hi = W + pad_w - off_cs;
    // prefetch
    bool checkal[TL] = ra_h >= h_lo && ra_h < h_hi && 
                       ra_w >= w_lo && ra_w < w_hi;
    bool checka[TM, TL] = checkal[newaxis, :];
    bool checkb[TL, TN] = 1;
#else
    // load LUT header
    int *header = lut + pid0 * 4;
    int a_offset = *(header + 0);
    int b_offset = *(header + 1);
    int L        = *(header + 2);
    int column   = *(header + 3);
    int lockid = 0;
    int maxid = 1;
    // initialize a pointers
    int rc_npq[TM]    = (pid1 * TM) + 0 ... TM;
    int rc_np [TM]    = rc_npq  / Q;
    int rc_q  [TM]    = rc_npq  % Q;
    int rc_p  [TM]    = rc_np   % P;
    int rc_n  [TM]    = rc_np   / P;
    int* pa_delta = lut + a_offset;
    int a_delta  __multipleof(TL) = *pa_delta;
    int ra_n  [TM]    = rc_n;
#ifdef DX
    int ra_h  [TM]    = rc_p * stride_h + pad_h;
    int ra_w  [TM]    = rc_q * stride_w + pad_w;
#else
    int ra_h  [TM]    = rc_p * stride_h - pad_h;
    int ra_w  [TM]    = rc_q * stride_w - pad_w;
#endif
    int ra_c  [TL]    = 0 ... TL;
    int offa[TM, TL]  = a_delta + ra_n[:, newaxis] * stride_na
                                + ra_h[:, newaxis] * stride_ha
                                + ra_w[:, newaxis] * stride_wa
                                + ra_c[newaxis, :] * 1;
    TYPE* pa[TM, TL]  = A + offa;
    // initialize b pointers
    int  rb_k[TN]     = 0 ... TN;
    int  rb_c[TL]     = 0 ... TL;
    int* pb_delta     = lut + b_offset;
    int  b_delta __multipleof(TL) = *pb_delta;
    TYPE* pb[TL, TN]  = B + b_delta + rb_k[newaxis, :] * STRIDE_BK
                                    + rb_c[:, newaxis] * STRIDE_BC;
    // prefetch
    int r = *(pa_delta + 1);
    int s = *(pa_delta + 2);
#ifdef DX
      int ra_hh[TM] = ra_h - r;
      int ra_ww[TM] = ra_w - s;
#else
      int ra_hh[TM] = ra_h + r;
      int ra_ww[TM] = ra_w + s;
#endif
    bool checkam[TM]    = ra_hh >= 0 && ra_hh < H && 
                          ra_ww >= 0 && ra_ww < W;
    bool checka[TM, TL] = checkam[:, newaxis];
    bool checkb[TL, TN] = 1;
#endif
    TYPE a[TM, TL] = checka ? *pa : 0;
    TYPE b[TL, TN] = checkb ? *pb : 0;

    /* ---------------- */
    /*    Inner Loop    */
    /* ---------------- */
    // create result tile
    float acc[TM, TN] = 0;
    for(int l = L; l > 0; l -= TL) {
      acc += a @ b;
#ifdef DW
      pa_delta += TL;
      a_delta = *pa_delta;
      pa += a_delta[newaxis, :];
      pb += TL * K;
      ra_nhw += TL;
      ra_nh   =  ra_nhw / Q;
      ra_w    = (ra_nhw % Q)*stride_w;
      ra_h    = (ra_nh  % P)*stride_h;
      bool checkal[TL] = ra_h >= h_lo && ra_h < h_hi && 
                         ra_w >= w_lo && ra_w < w_hi;
      bool checkal2[TL] = l > TL;
      bool checka[TM, TL] = checkal[newaxis, :] && checkal2[newaxis, :];
#else
      // update pointers
      pa_delta += 3;
      pb_delta += 1;
      int a_delta __multipleof(TL) = *pa_delta;
      int b_delta __multipleof(TL) = *pb_delta;
      pa += a_delta;
      pb += b_delta;
      int r = *(pa_delta + 1);
      int s = *(pa_delta + 2);
#ifdef DX
      int ra_hh[TM] = ra_h - r;
      int ra_ww[TM] = ra_w - s;
#else
      int ra_hh[TM] = ra_h + r;
      int ra_ww[TM] = ra_w + s;
#endif
      bool checkam[TM] = ra_hh >= 0 && ra_hh < H &&
                         ra_ww >= 0 && ra_ww < W; 
      bool checkal[TL] = l > TL;
      bool checka[TM, TL] = checkam[:, newaxis] && checkal[newaxis, :];
#endif
      // pre-fetch
      bool checkb[TL, TN] = l > TL;
      a = checka ? *pa : 0;
      b = *?(checkb)pb;
    }
    TYPE c[TM, TN] = acc;

    /* ---------------- */
    /*    Epilogue      */
    /* ---------------- */
    // initialize y pointers
#ifdef DW
    int  rc_c[TM] = 0 ... TM;
    int  rc_k[TN] = 0 ... TN;
    TYPE* pc[TM, TN] = C + pid0 * BLOCK * BLOCK
                         + rc_k[newaxis, :] * BLOCK
                         + rc_c[:, newaxis] * 1;
    bool checkc[TM, TN] = 1;
#else
    int rc_k[TN]     = column * TN + 0 ... TN;
    int offc[TM, TN] = rc_n[:, newaxis] * stride_nc
                         + rc_p[:, newaxis] * stride_hc
                         + rc_q[:, newaxis] * stride_wc
                         + rc_k[newaxis, :] * 1;
    TYPE* pc[TM, TN] = C + offc;
    bool checkc[TM, TN] = rc_npq[:, newaxis] < N*P*Q;
#endif
    // write-back directly
    if(lockid == 0) {
      *?(checkc) pc = c;
    }
    // accumulate partial result using spin-locks
    else {
      int *plock = locks + get_program_id(2)*nlocks*get_num_programs(1) + get_program_id(1)*nlocks + lockid - 1;
      int *pcount = plock + get_num_programs(2)*get_num_programs(1)*nlocks;
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % maxid);
      atomic_xchg(plock, 0);
    }
  }
'''

class _sparse_conv2d(torch.autograd.Function):

  _step = 16

  ##########################
  # UTILITIES              #
  ##########################
  locks = None
  @staticmethod
  def get_locks(size):
    if _sparse_conv2d.locks is None or \
        size > _sparse_conv2d.locks.size(0):
      _sparse_conv2d.locks = torch.zeros(size, dtype=torch.int32).cuda()
    return _sparse_conv2d.locks

  @staticmethod
  def make_dds_lut(layout, block, step, is_dx, strides, full_layout, off_bh, off_bw, stride_bh, stride_bw):
    headers  = torch.empty((0,), dtype=torch.int64)
    a_deltas = torch.empty((0,), dtype=torch.int64)
    b_deltas = torch.empty((0,), dtype=torch.int64) 
    a_deltas_start = 0
    width = 0
    div = block // step
    # pointer increments for b
    if is_dx:
      size = layout.sum()
      # blocks are stored in order KRSC
      block_id = full_layout.clone().permute(0, 2, 3, 1).contiguous()
      block_id[block_id > 0] = 1 + torch.arange(full_layout.sum())
      # blocks are traversed in order CRSK
      block_id = block_id.permute(3, 1, 2, 0).contiguous()
      block_id = block_id[:, off_bh::stride_bh, off_bw::stride_bw, :]
      b_offset = block_id[block_id > 0] - 1
      b_offset = b_offset * block * block
      b_deltas = b_offset.clone()
      b_deltas[1:] -= b_offset[:-1]
      # starting position in delta table
      sizes = torch.zeros(layout.shape[1], dtype=torch.int64)
      for c in range(layout.shape[1]):
        sizes[c] = layout[:, c, :, :].sum()
      b_deltas_start = torch.zeros(layout.shape[1], dtype=torch.int64)
      b_deltas_start[1:] = sizes.cumsum(0)[:-1]
      b_deltas[b_deltas_start] = b_offset[b_deltas_start]
    else:
      b_offset = torch.arange(layout.sum())
      b_offset = b_offset * block * block
      b_deltas = b_offset.clone()
      b_deltas[1:] -= b_offset[:-1]
      b_deltas = b_deltas.view(-1)
      b_deltas_start = torch.zeros(layout.shape[0], dtype=torch.int64)
      b_deltas_start[1:] = layout.view(layout.shape[0], -1).sum(1).cumsum(0)[:-1]
      b_deltas[b_deltas_start] = b_offset[b_deltas_start]
    # handle step
    b_deltas = b_deltas.view(-1, 1).repeat(1, div)
    if not is_dx:
      b_deltas[:, 1:] = step
      b_deltas[:, 0] -= (div-1)*step
    else:
      b_deltas[:, 1:] = step*block
      b_deltas[:, 0] -= (div - 1)*step*block
    b_deltas[b_deltas_start, 0] = b_offset[b_deltas_start]
    b_deltas = b_deltas.view(-1)
    b_deltas_start *= div
    # headers and pointer increments for a
    out_dim = 1 if is_dx else 0
    for k in range(layout.shape[out_dim]):
        if is_dx:
          nnz = layout[:, k, :, :].permute(1, 2, 0).nonzero()
          a_coffset = nnz[:,2]*block*strides[0] - \
                      nnz[:,1]*strides[1] - \
                      nnz[:,0]*strides[2]
          a_noffset = nnz[1:,2]*block*strides[0] - \
                      nnz[1:,1]*strides[1] - \
                      nnz[1:,0]*strides[2]
        else:
          nnz = layout[k, :, :, :].permute(1, 2, 0).nonzero()
          a_coffset = nnz[:,2]*block*strides[0] + \
                      nnz[:,1]*strides[1] + \
                      nnz[:,0]*strides[2]
          a_noffset = nnz[1:,2]*block*strides[0] + \
                      nnz[1:,1]*strides[1] + \
                      nnz[1:,0]*strides[2]
        a_inc  = a_noffset - a_coffset[:-1]
        a_inc  = torch.cat((a_coffset[:1], a_inc))
        # handle step
        offset = a_inc[0]
        a_inc = a_inc.view(-1, 1).repeat(1, div)
        a_inc[:, 1:] = step
        a_inc[:, 0] -= (div - 1)*step
        a_inc = a_inc.view(-1)
        a_inc[0] = offset
        # filter indices
        a_rr  = nnz[:, 0].view(-1, 1).repeat(1, div).view(-1)
        a_ss  = nnz[:, 1].view(-1, 1).repeat(1, div).view(-1)
        # build look-up table
        a_dd  = torch.stack((a_inc, a_rr, a_ss), dim=1).view(-1).contiguous()
        a_deltas = torch.cat((a_deltas, a_dd))
        # create headers
        size = nnz.shape[0]*div
        hh = torch.tensor([a_deltas_start, b_deltas_start[k], size*step, k], dtype=torch.int64)
        a_deltas_start += 3*size
        headers = torch.cat((headers, hh))
        # update width
        width += 1
    # create look-up table
    headers[0::4] += headers.shape[0]
    headers[1::4] += headers.shape[0] + a_deltas.shape[0]
    lut = torch.cat((headers, a_deltas, b_deltas)).type(torch.int32).cuda()
    num_locks = 1
    return lut, num_locks, width
  
  @staticmethod
  def make_sdd_lut(layout, block):
    nnz = layout.permute(0, 2, 3, 1).contiguous().nonzero()
    width = layout.sum()
    # create lut
    k = nnz[:, 0]
    r = nnz[:, 1]
    s = nnz[:, 2]
    c = nnz[:, 3]
    lut = torch.stack((k, c, r, s), dim=1).view(-1).contiguous()
    lut = lut.type(torch.int32).cuda()
    # create locks
    num_locks = 1
    return lut, num_locks, width

  @staticmethod
  def unpack(idx, N, H, W):
    w  = idx % W
    nh = idx // W
    h  = nh % H
    n  = nh // H
    return n, h, w

  @staticmethod
  def make_db_delta(N, H, W, stride_n, stride_h, stride_w, step, 
                    transform_h = lambda h: h,
                    transform_w = lambda w: w):
    # aggregate reduction indices
    idx = torch.arange(N*H*W, dtype=torch.int32)
    next_idx = idx + step
    # unpacked reduction indices
    n, h, w = _sparse_conv2d.unpack(idx, N, H, W)
    next_n, next_h, next_w = _sparse_conv2d.unpack(next_idx, N, H, W)
    # transform indices
    h, next_h = transform_h(h), transform_h(next_h)
    w, next_w = transform_w(w), transform_w(next_w)
    # memory addresses
    off = w * stride_w + h * stride_h + n * stride_n
    next_off = next_w * stride_w + next_h * stride_h + next_n * stride_n
    # deltas
    ret = torch.cat((off[:step], next_off - off)).cuda()
    return ret
    

  sdd_cache = dict()
  dds_cache = dict()
  @staticmethod
  def make_kernel(src, defines, cache, key, num_warps=[4]):
    if key not in cache:
      cache[key] = triton.kernel(src, defines=defines, num_warps=num_warps)
    return cache[key]

  ##########################
  # OPERATORS              #
  ##########################

  # Sparse = Dense x Dense
  @staticmethod
  def _sdd_conv2d(a, b, pad_h, pad_w, stride_h, stride_w,
                  layout, block, lut, num_locks, width, 
                  bench, time):
    # sanity checks
    a_dtype = a.dtype
    b_dtype = b.dtype
    Na, C, H, W = a.shape
    Nb, K, P, Q = b.shape
    _, _, R, S = layout.shape
    assert a_dtype == b_dtype
    assert Na == Nb
    # create kernel
    defines = {'NAME': 'sdd_conv2d', 'TYPE': a.dtype,
               'TM': block, 'TL': 16, 'TN': block, 'BLOCK': block,
               'TZ': 1, 'DW': True}
    cache = _sparse_conv2d.sdd_cache
    kernel = _sparse_conv2d.make_kernel(src, defines, cache, (block, a_dtype), num_warps=[2])
    # create semaphores
    locks = _sparse_conv2d.get_locks(2*width*num_locks)
    # create output
    c = torch.empty((layout.sum(), block, block), dtype=a.dtype, device=a.device)
    kernel(a, b, c, 
          H, W, R, S, C,
          Na, P, Q, K,
          pad_h, pad_w, stride_h, stride_w,
          a.stride(0), a.stride(2), a.stride(3),
          b.stride(0), b.stride(3), b.stride(3),
          lut, locks, num_locks, 
          grid = lambda opt: [width, opt.d('TZ')], 
          bench = bench)
    return c

  @staticmethod
  def pad(tensor, pad):
      pad = pad + [0] *  (2*len(tensor.shape) - len(pad))
      begin = [ x if x > 0 else None for x in pad[-1::-2]]
      end   = [-x if x > 0 else None for x in pad[-2::-2]]
      slices = [slice(b, e) for b, e in zip(begin, end)]
      tensor = torch.nn.functional.pad(tensor, pad, 'constant', 0).to(memory_format=torch.channels_last)
      tensor = tensor[slices]
      return tensor

  # Dense = Dense x Sparse
  @staticmethod
  def _dds_conv2d(a, b, nchwkrspq,
                  pad_h, pad_w, stride_h, stride_w,
                  is_dx, layout, block, 
                  lut, num_locks, width, da_offs,
                  bench, time):
    N, C, H, W, K, R, S, P, Q = nchwkrspq
    # swap shapes
    if is_dx:
      C, K = K, C
      H, P = P, H
      W, Q = Q, W
    # create kernel
    defines = {'NAME': 'dds_conv2d_' + ('_dx' if is_dx else '_y'), 'TYPE': a.dtype,
               'TM': 128, 'TL': _sparse_conv2d._step, 'TN': block, 'BLOCK': block,
               'STRIDE_BK': 1 if is_dx else block,
               'STRIDE_BC': block if is_dx else 1}
    if is_dx:
      defines['DX'] = True
    cache = _sparse_conv2d.dds_cache
    kernel = _sparse_conv2d.make_kernel(src, defines, cache, (block, a.dtype, is_dx), num_warps=[4])
    # create output
    c = torch.empty((N, K, P, Q), dtype=a.dtype, device=a.device).contiguous(memory_format=torch.channels_last)
    if is_dx:
      for da_lut, da_num_locks, da_width, (a_pad_h, a_pad_w, off_bh, off_bw, off_ch, off_cw) in zip(lut, num_locks, width, da_offs):
        if lut is None:
          c[:, :, offh::stride_h, offw::stride_w] = 0
        else:
          da_locks = _sparse_conv2d.get_locks(2*da_width*da_num_locks*N*P*Q)
          cc = c[:, :, off_ch::stride_h, off_cw::stride_w]
          N, K, P, Q = cc.shape
          kernel(a, b, cc,
                H, W, R, S, C,
                N, P, Q, K,
                a_pad_h, a_pad_w, 
                1, 1,
                a.stride(0), a.stride(2), a.stride(3),
                cc.stride(0), cc.stride(2), cc.stride(3),
                da_lut, da_locks, da_num_locks, 
                grid = lambda opt: [da_width, triton.cdiv(N*P*Q, opt.d('TM'))], 
                bench = bench)
    else:
      locks = _sparse_conv2d.get_locks(2*width*num_locks*N*P*Q)
      kernel(a, b, c, 
            H, W, R, S, C,
            N, P, Q, K,
            pad_h, pad_w, stride_h, stride_w,
            a.stride(0), a.stride(2), a.stride(3),
            c.stride(0), c.stride(2), c.stride(3),
            lut, locks, num_locks, 
            grid = lambda opt: [width, triton.cdiv(N*P*Q, opt.d('TM'))], 
            bench = bench)

    return c

  
  @staticmethod
  def forward(ctx, a, b, 
              nchwkrspq, pad_h, pad_w, stride_h, stride_w, 
              layout, block,
              c_lut,  c_num_locks,  c_width,
              da_lut, da_num_locks, da_width, da_offs,
              db_lut, db_num_locks, db_width,
              bench, c_time, da_time, db_time):
    c = _sparse_conv2d._dds_conv2d(a, b, 
                                   nchwkrspq, pad_h, pad_w, stride_h, stride_w,
                                   False, layout, block, 
                                   c_lut, c_num_locks, c_width, None,
                                   bench, c_time)
    # save for backward
    ctx.save_for_backward(a, b)
    # da parameters
    ctx.da_lut = da_lut
    ctx.da_num_locks = da_num_locks
    ctx.da_width = da_width
    ctx.da_time = da_time
    # db parameters
    ctx.db_lut = db_lut
    ctx.db_num_locks = db_num_locks
    ctx.db_width = db_width
    ctx.db_time = db_time
    # conv parameters
    ctx.nchwkrspq = nchwkrspq
    ctx.bench = bench
    ctx.block = block
    ctx.layout = layout
    ctx.pad_h = pad_h
    ctx.pad_w = pad_w
    ctx.stride_h = stride_h
    ctx.stride_w = stride_w
    ctx.da_offs = da_offs
    return c
  
  @staticmethod
  def backward(ctx, dc):
    # retrieve from context
    a, b         = ctx.saved_tensors
    da_lut       = ctx.da_lut
    da_num_locks = ctx.da_num_locks 
    da_width     = ctx.da_width 
    da_time      = ctx.da_time
    da_offs      = ctx.da_offs
    db_lut       = ctx.db_lut
    db_num_locks = ctx.db_num_locks
    db_width     = ctx.db_width
    db_time      = ctx.db_time
    bench        = ctx.bench
    block        = ctx.block
    layout       = ctx.layout
    pad_h        = ctx.pad_h
    pad_w        = ctx.pad_w
    stride_h     = ctx.stride_h
    stride_w     = ctx.stride_w
    nchwkrspq    = ctx.nchwkrspq
    # gradients w.r.t. a
    da = None
    if ctx.needs_input_grad[0]:
      da = _sparse_conv2d._dds_conv2d(dc, b, nchwkrspq, pad_h, pad_w, stride_h, stride_w,
                       True, layout, block, 
                       da_lut, da_num_locks, da_width, da_offs,
                       bench, da_time)
    # gradients w.r.t. b
    db = None
    if ctx.needs_input_grad[1]:
      db = _sparse_conv2d._sdd_conv2d(a, dc, pad_h, pad_w, stride_h, stride_w,
                                      layout, block,
                                      db_lut, db_num_locks, db_width,
                                      bench, db_time)
    return da, db, None, None,\
           None, None, None, None, None,\
           None, None, None,\
           None, None, None, None,\
           None, None, None,\
           None, None, None, None


class SparseConv2d:

  sparse_conv2d = _sparse_conv2d.apply

  def __init__(self, layout, block, N, C, H, W, P, Q, K, R, S, stride_h, stride_w, pad_h, pad_w):
    # attributes
    self.layout = layout
    self.block = block
    self.nchwkrspq = N, C, H, W, K, R, S, P, Q

    # Look-up tables for forward pass
    self.c_lut,  self.c_num_locks,  self.c_width  = _sparse_conv2d.make_dds_lut(layout, block, _sparse_conv2d._step, False, [1, C, C*W], None, None, None, None, None)
    # Look-up tables for data gradient
    # have to be careful here
    # the gradient of strided conv is a conv over a sparse image
    # which can be decomposed as a set of smaller convs
    #da_pad_h = (H - P*stride_h + R - 1)//2
    #da_pad_w = (W - Q*stride_w + S - 1)//2
    #print(da_pad_h, da_pad_w)
    self.da_lut, self.da_num_locks, self.da_width = [], [], []
    self.da_offs = []
    for off_ch in range(stride_h):
      for off_cw in range(stride_w):
        off_bh = (off_ch + pad_h) % stride_h
        off_bw = (off_cw + pad_w) % stride_w
        a_pad_h = int((pad_h + (stride_h - 1)*off_ch) / stride_h)
        a_pad_w = int((pad_w + (stride_w - 1)*off_cw) / stride_w)
        if off_bh >= R or off_bw >= S:
          da_lut, da_num_locks, da_width = None
        else:
          curr_layout = layout[:, :, off_bh::stride_h, off_bw::stride_w]
          da_lut, da_num_locks, da_width = _sparse_conv2d.make_dds_lut(curr_layout, block, _sparse_conv2d._step, True, [1, K, K*Q], layout, off_bh, off_bw, stride_h, stride_w)
        self.da_lut.append(da_lut)
        self.da_num_locks.append(da_num_locks)
        self.da_width.append(da_width)
        self.da_offs.append((a_pad_h, a_pad_w, off_bh, off_bw, off_ch, off_cw))
        
    # look-up tables for weight gradients
    self.db_lut, self.db_num_locks, self.db_width = _sparse_conv2d.make_sdd_lut(layout, block)
    db_delta_a = _sparse_conv2d.make_db_delta(N, P, Q, C*W*H, C*W, C, 16,
                                              transform_h = lambda h: h*stride_h - pad_h,
                                              transform_w = lambda w: w*stride_w - pad_w)
    db_delta_b = _sparse_conv2d.make_db_delta(N, P, Q, K*Q*P, K*Q, K, 16)
    self.db_lut = torch.cat((self.db_lut, db_delta_a, db_delta_b))

    # timings
    self.bench = False
    self.c_time = [None]
    self.da_time = [None]
    self.db_time = [None]

  def __call__(self, a, b, pad_h, pad_w, stride_h, stride_w):
    c = SparseConv2d.sparse_conv2d(a, b, 
                                  self.nchwkrspq, pad_h, pad_w, stride_h, stride_w,
                                  self.layout, self.block,
                                  self.c_lut, self.c_num_locks, self.c_width,
                                  self.da_lut, self.da_num_locks, self.da_width, self.da_offs,
                                  self.db_lut, self.db_num_locks, self.db_width,
                                  self.bench, self.c_time, self.da_time, self.db_time)
    return c