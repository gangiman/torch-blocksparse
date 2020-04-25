import triton
import torch
from .matmul import _sparse_matmul
import math

src = '''
  __global__ void NAME (TYPE* A __readonly  __noalias __aligned(16),
                        TYPE* B __readonly  __noalias __aligned(16),
                        TYPE* C __noalias __aligned(16),
                        // shapes
                        int N, int P, int Q, int K,
                        // a strides
                        int stride_na __multipleof(8),
                        int stride_ha __multipleof(8),
                        int stride_wa __multipleof(8),
                        // c strides
                        int stride_nc __multipleof(8),
                        int stride_hc __multipleof(8),
                        int stride_wc __multipleof(8),
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
    int L = N*P*Q;
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
    int* pa_delta[TL] = lut + a_offset + 0 ... TL;
    int a_delta[TL]   = *pa_delta;
    TYPE* pa[TM, TL]  = A + rc_n[:, newaxis] * stride_na
                          + rc_p[:, newaxis] * stride_ha
                          + rc_q[:, newaxis] * stride_wa
                          + a_delta[newaxis, :];
    // initialize b pointers
    int  rb_k[TN]     = 0 ... TN;
    int  rb_c[TL]     = 0 ... TL;
    int* pb_delta     = lut + b_offset;
    int  b_delta      = *pb_delta;
    TYPE* pb[TL, TN]  = B + b_delta + rb_k[newaxis, :] * STRIDE_BK
                                    + rb_c[:, newaxis] * STRIDE_BC;
#endif
    // prefetch
    bool checka[TM, TL] = 1;
    bool checkb[TL, TN] = 1;
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
#else
      // update pointers
      pa_delta += TL;
      pb_delta += 1;
      a_delta = *pa_delta;
      b_delta = *pb_delta;
      pa += a_delta[newaxis, :];
      pb += b_delta;
#endif
      // pre-fetch
      bool checka[TM, TL] = l > TL;
      bool checkb[TL, TN] = l > TL;
      a = *?(checka)pa;
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
    TYPE* pc[TM, TN] = C + rc_n[:, newaxis] * stride_nc
                         + rc_k[newaxis, :] * 1
                         + rc_p[:, newaxis] * stride_hc
                         + rc_q[:, newaxis] * stride_wc;
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
  def make_dds_lut(layout, block, step, is_dx, strides):
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
      block_id = layout.clone().permute(0, 2, 3, 1).contiguous()
      block_id[block_id > 0] = 1 + torch.arange(size)
      # blocks are traversed in order CRSK
      block_id = block_id.permute(3, 1, 2, 0).contiguous()
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
    # headers and pointer increments for a
    out_dim = 1 if is_dx else 0
    for k in range(layout.shape[out_dim]):
        if is_dx:
          repeats = block*torch.ones(layout.shape[0], dtype=torch.int64)
          nnz = layout[:, k, :, :].repeat_interleave(repeats, dim=0).permute(1, 2, 0).nonzero()
          a_coffset = nnz[:,2]*strides[0] - \
                      nnz[:,1]*strides[1] - \
                      nnz[:,0]*strides[2]
          a_noffset = nnz[step:,2]*strides[0] - \
                      nnz[step:,1]*strides[1] - \
                      nnz[step:,0]*strides[2]
        else:
          repeats = block*torch.ones(layout.shape[1], dtype=torch.int64)
          nnz = layout[k, :, :, :].repeat_interleave(repeats, dim=0).permute(1, 2, 0).nonzero()
          a_coffset = nnz[:,2]*strides[0] + \
                      nnz[:,1]*strides[1] + \
                      nnz[:,0]*strides[2]
          a_noffset = nnz[step:,2]*strides[0] + \
                      nnz[step:,1]*strides[1] + \
                      nnz[step:,0]*strides[2]
        a_dd  = a_noffset - a_coffset[:-step]
        a_dd  = torch.cat((a_coffset[:step], a_dd))
        a_deltas = torch.cat((a_deltas, a_dd))
        # create headers
        size = a_dd.shape[0]
        hh = torch.tensor([a_deltas_start, b_deltas_start[k], size, k], dtype=torch.int64)
        a_deltas_start += size
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
  def make_db_delta(N, H, W, stride_n, stride_h, stride_w, step):
    # aggregate reduction indices
    idx = torch.arange(N*H*W, dtype=torch.int32)
    next_idx = idx + step
    # unpacked reduction indices
    n, h, w = _sparse_conv2d.unpack(idx, N, H, W)
    next_n, next_h, next_w = _sparse_conv2d.unpack(next_idx, N, H, W)
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
  def _sdd_conv2d(a, b,
                  layout, block, lut, num_locks, width, 
                  bench, time):
    # sanity checks
    a_dtype = a.dtype
    b_dtype = b.dtype
    Na, C, H, W = a.shape
    Nb, K, P, Q = b.shape
    assert a_dtype == b_dtype
    assert Na == Nb
    # create kernel
    defines = {'NAME': 'sdd_conv2d', 'TYPE': a.dtype,
               'TM': block, 'TL': 8, 'TN': block, 'BLOCK': block,
               'TZ': 1, 'DW': True}
    cache = _sparse_conv2d.sdd_cache
    kernel = _sparse_conv2d.make_kernel(src, defines, cache, (block, a_dtype))
    # create semaphores
    locks = _sparse_conv2d.get_locks(2*width*num_locks)
    # create output
    c = torch.empty((layout.sum()*block, block), dtype=a.dtype, device=a.device)
    kernel(a, b, c, 
          Na, P, Q, K,
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
  def _dds_conv2d(a, b, is_dx, layout, block, 
                  lut, num_locks, width,
                  bench, time):
    # sanity checks
    N, Ca, H, W = a.shape
    K, Cb, R, S = layout.shape[0]*block, layout.shape[1]*block,\
                  layout.shape[2], layout.shape[3]
    if is_dx:
      Cb, K = K, Cb
      P = H + R - 1
      Q = W + S - 1
    else:
      P = H - R + 1
      Q = W - S + 1
    assert a.dtype == b.dtype
    assert Ca == Cb
    # pad
    if is_dx:
      a = _sparse_conv2d.pad(a, [4, 4, 4, 4])
    # create kernel
    defines = {'NAME': 'dds_conv2d', 'TYPE': a.dtype,
               'TM': 32, 'TL': 16, 'TN': block, 'BLOCK': block,
               'STRIDE_BK': 1 if is_dx else block,
               'STRIDE_BC': block if is_dx else 1}
    cache = _sparse_conv2d.dds_cache
    kernel = _sparse_conv2d.make_kernel(src, defines, cache, (block, a.dtype, is_dx))
    # create semaphores
    locks = _sparse_conv2d.get_locks(2*width*num_locks*N*P*Q)
    # create output
    c = torch.empty((N, K, P, Q), dtype=a.dtype, device=a.device).contiguous(memory_format=torch.channels_last)
    kernel(a, b, c, 
          N, P, Q, K,
          a.stride(0), a.stride(2), a.stride(3),
          c.stride(0), c.stride(2), c.stride(3),
          lut, locks, num_locks, 
          grid = lambda opt: [width, triton.cdiv(N*P*Q, opt.d('TM'))], 
          bench = bench)
    return c

  
  @staticmethod
  def forward(ctx, a, b, layout, block,
              c_lut,  c_num_locks,  c_width,
              da_lut, da_num_locks, da_width,
              db_lut, db_num_locks, db_width,
              bench, c_time, da_time, db_time):
    c = _sparse_conv2d._dds_conv2d(a, b, False, layout, block, 
                                   c_lut, c_num_locks, c_width, 
                                   bench, c_time)
    # save for backward
    ctx.save_for_backward(a, da_lut, b, db_lut)
    ctx.da_num_locks = da_num_locks
    ctx.da_width = da_width
    ctx.da_time = da_time
    ctx.db_num_locks = db_num_locks
    ctx.db_width = db_width
    ctx.db_time = db_time
    ctx.bench = bench
    ctx.block = block
    ctx.layout = layout
    return c
  
  @staticmethod
  def backward(ctx, dc):
    # retrieve from context
    a, da_lut, b, db_lut = ctx.saved_tensors
    da_num_locks = ctx.da_num_locks 
    da_width     = ctx.da_width 
    da_time      = ctx.da_time
    db_num_locks = ctx.db_num_locks
    db_width     = ctx.db_width
    db_time      = ctx.db_time
    bench        = ctx.bench
    block        = ctx.block
    layout       = ctx.layout
    # gradients w.r.t. a
    da = None
    if ctx.needs_input_grad[0]:
      da = _sparse_conv2d._dds_conv2d(dc, b, True, layout, block, 
                       da_lut, da_num_locks, da_width, 
                       bench, da_time)
    # gradients w.r.t. b
    db = None
    if ctx.needs_input_grad[1]:
      db = _sparse_conv2d._sdd_conv2d(a, dc, layout, block,
                                      db_lut, db_num_locks, db_width,
                                      bench, db_time)
    return da, db, None, None,\
           None, None, None,\
           None, None, None,\
           None, None, None,\
           None, None, None, None


class SparseConv2d:

  sparse_conv2d = _sparse_conv2d.apply

  def __init__(self, layout, block, N, C, H, W, P, Q, K):
    # attributes
    self.layout = layout
    self.block = block
    # look-up tables
    QQ, PP = Q + 8, P + 8
    # assume NHWC format
    self.c_lut,  self.c_num_locks,  self.c_width  = _sparse_conv2d.make_dds_lut(layout, block, 16, False, [1, C, C*W])
    self.da_lut, self.da_num_locks, self.da_width = _sparse_conv2d.make_dds_lut(layout, block, 16, True,  [1, K, K*QQ])
    self.db_lut, self.db_num_locks, self.db_width = _sparse_conv2d.make_sdd_lut(layout, block)
    db_delta_a = _sparse_conv2d.make_db_delta(N, P, Q, C*W*H, C*W, C, 8)
    db_delta_b = _sparse_conv2d.make_db_delta(N, P, Q, K*Q*P, K*Q, K, 8)
    self.db_lut = torch.cat((self.db_lut, db_delta_a, db_delta_b))
    # timings
    self.bench = False
    self.c_time = [None]
    self.da_time = [None]
    self.db_time = [None]

  def __call__(self, a, b):
    c = SparseConv2d.sparse_conv2d(a, b, self.layout, self.block,
                                  self.c_lut, self.c_num_locks, self.c_width,
                                  self.da_lut, self.da_num_locks, self.da_width,
                                  self.db_lut, self.db_num_locks, self.db_width,
                                  self.bench, self.c_time, self.da_time, self.db_time)
    return c