from torch_blocksparse import *
import torch
torch.manual_seed(0)

############
## UTILS  ##
############

# convert dense matrix with explicit zeros to sparse matrix
def dense_to_sparse(w, mask, block):
  Z = w.size(0)
  ret = torch.empty((Z, mask.sum(), block, block), dtype=w.dtype, device=w.device)
  nnz = mask.nonzero()
  h, i, j = nnz[:, 0], nnz[:, 1], nnz[:, 2]
  for zz in range(Z):
    for idx, (hh, ii, jj) in enumerate(zip(h, i, j)):
      ret[zz, idx, :, :] = w[zz, hh, ii*block: (ii+1)*block, jj*block: (jj+1)*block]
  return ret

# convert sparse matrix to dense matrix with explicit zeros
def sparse_to_dense(w, mask, block, zero = 0):
  maskedw = w.clone()
  for bz, wz in enumerate(range(0, w.size(0))):
    for bh, wh in enumerate(range(0, w.size(1))):
      for bi, wi in enumerate(range(0, w.size(2), block)):
        for bj, wj in enumerate(range(0, w.size(3), block)):
          if mask[bh, bi, bj] == 0:
            maskedw[wz, wh, wi : wi+block, wj:wj+block] = zero
          #maskedw[wz, wh, wi : wi+block, wj : wj+block] *= mask[bh, bi, bj]
  return maskedw

##########
# MatMul #
##########

# run reference implementation
def run_mm_reference(x, w, mode, trans_a, trans_b, layout, block, dy):
  x = sparse_to_dense(x, layout, block) if mode == 'dsd' else x
  w = sparse_to_dense(w, layout, block) if mode == 'dds' else w
  x.retain_grad()
  w.retain_grad()
  xx = x.transpose(2, 3) if trans_a else x
  ww = w.transpose(2, 3) if trans_b else w
  y = torch.matmul(xx, ww)
  y = sparse_to_dense(y, layout, block) if mode == 'sdd' else y
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  w.grad.zero_()
  y = dense_to_sparse(y, layout, block) if mode == 'sdd' else y
  dx = dense_to_sparse(dx, layout, block) if mode == 'dsd' else dx
  dw = dense_to_sparse(dw, layout, block) if mode == 'dds' else dw
  return y, dx, dw

# run triton implementation
def run_mm_triton(x, w, mode, trans_a, trans_b, layout, block, dy):
  x = dense_to_sparse(x, layout, block) if mode == 'dsd' else x
  w = dense_to_sparse(w, layout, block) if mode == 'dds' else w
  dy = dense_to_sparse(dy, layout, block) if mode == 'sdd' else dy
  op = SparseMatMul(layout, block, mode, trans_a=trans_a, trans_b=trans_b)
  x.retain_grad()
  w.retain_grad()
  y = op(x, w)
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  return y, dx, dw

# benchmark triton implementation
def bench_mm_triton(x, w, mode, trans_a, trans_b, layout, block, num_repeat):
  from time import time
  x = dense_to_sparse(x, layout, block) if mode == 'dsd' else x
  w = dense_to_sparse(w, layout, block) if mode == 'dds' else w
  op = SparseMatMul(layout, block, mode, trans_a=trans_a, trans_b=trans_b)
  op.bench = num_repeat
  y = op(x, w)
  torch.cuda.synchronize()
  y = op(x, w)
  torch.cuda.synchronize()
  return op.time_c*1e-9
  
def bench_mm_openai(x, w, mode, trans_a, trans_b, layout, block, num_repeat):
  # import and disable all logging
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  import warnings
  warnings.filterwarnings('ignore',category=FutureWarning)
  from blocksparse.matmul import BlocksparseMatMul
  from blocksparse.transformer import BlocksparseTransformer
  import tensorflow as tf
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  import numpy as np
  sparsity = layout.cpu().numpy()
  # create operator
  transformer = BlocksparseTransformer(sparsity, heads=layout.shape[0], block_size=block)
  dot_sdd_nt = transformer.nt_op
  dot_dsd_tn = transformer.tn_op
  dot_dsd_nn = transformer.nn_op
  dot_dds_nn = None if mode != 'dds' else BlocksparseMatMul(sparsity[0,:,:], block_size=block)
  key = (mode, trans_a, trans_b)
  ops = {('sdd', False, True): dot_sdd_nt,
         ('dsd', True, False): dot_dsd_tn,
         ('dsd', False, False): dot_dsd_nn,
         ('dds', False, False): dot_dds_nn}
  if x.dtype == torch.float32 and (mode == 'dsd' or block != 32):
    return None
  if key not in ops:
    return None
  if mode == 'dds' and x.shape[0]*x.shape[1] != 1:
    return None
  op = ops[key]
  # placeholder
  x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
  w = w.view(w.shape[0]*w.shape[1], w.shape[2], w.shape[3])
  sparse_shape = [x.shape[0], layout.shape[0], layout[0].sum(), block, block]
  vx = tf.placeholder(tf.float32, shape = sparse_shape if mode == 'dsd' else x.shape)
  vw = tf.placeholder(tf.float32, shape = sparse_shape if mode == 'dds' else w.shape)
  x = np.random.rand(*sparse_shape) if mode == 'dsd' else x.cpu().detach().numpy()
  w = np.random.rand(*sparse_shape) if mode == 'dds' else w.cpu().detach().numpy()
  # Block-sparse matrix multiplication
  y = op(vx, vw, bench=num_repeat)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([y], feed_dict = {vx: x, vw: w})
  sess.close()

def test_mm(Z, H, M, N, K, rho, mode, trans_a, trans_b, block):
  torch.manual_seed(1)
  AS0 = K if trans_a else M
  AS1 = M if trans_a else K
  BS0 = N if trans_b else K
  BS1 = K if trans_b else N
  shape = {'sdd': (M, N),
           'dsd': (AS0, AS1),
           'dds': (BS0, BS1)}[mode]
  # initialize layout
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  layout = generator.sample((H, shape[0]//block, shape[1]//block))
  layout[:] = layout[0, :, :]
  x = torch.rand((Z, H, AS0, AS1), dtype=torch.float32, requires_grad=True).cuda()
  w = torch.rand((Z, H, BS0, BS1), dtype=torch.float32, requires_grad=True).cuda()
  dy = torch.rand((Z, H, M, N), dtype=torch.float32).cuda()
  x.retain_grad()
  w.retain_grad()
  # run
  ry, rdx, rdw = run_mm_reference(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, dy)
  ty, tdx, tdw = run_mm_triton(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, dy)
  # test
  idx = (tdx - rdx).abs() > 1
  assert(torch.allclose(ty, ry))
  assert(torch.allclose(tdx, rdx))
  assert(torch.allclose(tdw, rdw))
  # benchmark
  num_repeat = 100
  triton_ts = bench_mm_triton(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, num_repeat)
  #openai_ts = bench_mm_openai(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, num_repeat)
  #flops = 2 * M * bsz * bsz * layout.sum()
  print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

###########
# Softmax #
###########

def run_softmax_triton(x, scale, dx, mask, layout, block):
  sparse_softmax = softmax.SparseSoftmax(layout, block, bench=False)
  dx = dense_to_sparse(dx, layout, block)
  x = dense_to_sparse(x, layout, block)
  x.retain_grad()
  y = sparse_softmax(x, scale=scale, mask=mask)
  y.backward(dx)
  dx = x.grad.clone()
  x.grad.zero_()
  return x, dx

def run_softmax_reference(x, scale, dx, mask, layout, block):
  x = sparse_to_dense(x, layout, block, zero=float('-inf'))
  x.retain_grad()
  if mask is not None:
    y = torch.softmax(x*scale + mask[:, None, None, :], -1)
  else:
    y = torch.softmax(x*scale, -1)
  y.backward(dx)
  dx = x.grad.clone()
  dx = dense_to_sparse(dx, layout, block)
  y = dense_to_sparse(y, layout, block)
  return y, dx
  
def bench_softmax_triton(x, scale, mask, layout, block):
  sparse_softmax = softmax.SparseSoftmax(layout, block, bench=True)
  x = dense_to_sparse(x, layout, block)
  x = sparse_softmax(x, scale=scale, mask=mask)
  return sparse_softmax.time_y*1e-9


def test_softmax(Z, H, M, N, scale, rho, block):
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # initialize tensors
  layout = generator.sample((H, M//block, N//block))
  x = torch.rand((Z, H, M, N), dtype=torch.float32, requires_grad=True).cuda()
  dx = torch.rand_like(x)
  mask = torch.randint(low=0, high=1, size=(Z, N), dtype=torch.float32, requires_grad=False).cuda()
  mask[mask==1.] = float('-inf')
  # execute
  ry, rdx = run_softmax_reference(x, scale, dx, mask, layout, block)
  ty, tdx = run_softmax_triton(x, scale, dx, mask, layout, block)
  assert(torch.allclose(ry, ty))
  assert(torch.allclose(rdx, tdx))
  # benchmark
  triton_ts = bench_softmax_triton(x, scale, mask, layout, block) 
  print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

###########
# CONV    #
###########

def mask_weights(w, layout, block):
  repeat_k = block*torch.ones(layout.shape[0], dtype=torch.int64)
  repeat_c = block*torch.ones(layout.shape[1], dtype=torch.int64)
  mask = layout.repeat_interleave(repeat_k, dim=0)\
                 .repeat_interleave(repeat_c, dim=1).cuda()
  return w * mask

def compress_weights(w, layout, block):
  blocks = torch.empty((0,), dtype=torch.float32)
  repeat_c = block*torch.ones(layout.shape[1], dtype=torch.int64)
  layout = layout.repeat_interleave(repeat_c, dim=1)
  for k in range(layout.shape[0]):
    nnz = layout[k, :, :, :].nonzero()
    compressed = w[k*block : (k+1)*block,
                   nnz[:,0], nnz[:,1], nnz[:, 2]].cpu()
    for bc in range(0, compressed.shape[1], block):
      blocks = torch.cat((blocks, compressed[:, bc:bc+block]))
  return blocks.cuda()

def run_conv2d_reference(x, w, dy, layout, block):
  # create conv2d
  C, K, R, S = x.shape[1], dy.shape[1], layout.shape[2], layout.shape[3]
  conv2d = torch.nn.Conv2d(w.shape[1], w.shape[0], (R, S), bias=False).cuda()
  conv2d.weight.data.copy_(mask_weights(w, layout, block))
  # run conv2d
  y = conv2d(x)
  # backward
  y.backward(dy)
  dx = x.grad.clone()
  dw = conv2d.weight.grad.clone()
  dw = compress_weights(dw, layout, block)
  x.grad.zero_()
  return y, dx, dw

def run_conv2d_triton(x, w, dy, layout, block):
  # create conv2d
  N, C, H, W = x.shape
  K = dy.shape[1]
  sparse_conv2d = SparseConv2d(layout, block, N, C, H, W, dy.shape[2], dy.shape[3], K)
  # run conv2d
  w = compress_weights(w, layout, block)
  w.retain_grad()
  y = sparse_conv2d(x, w)
  # backward
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  return y, dx, dw

def test_conv2d(N, C, H, W, K, R, S, rho, block):
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # initialize tensors
  layout = generator.sample((K//block, C//block, R, S))
  x = torch.rand((N, C, H, W), dtype=torch.float32, requires_grad=True).cuda()
  w = torch.ones((K, C, R, S), dtype=torch.float32, requires_grad=True).cuda()
  dy = torch.rand((N, K, H - R + 1, W - S + 1), dtype=torch.float32).cuda()
  x.retain_grad()
  w.retain_grad()
  # execute
  ry, rdx, rdw = run_conv2d_reference(x, w, dy, layout, block)
  ty, tdx, tdw = run_conv2d_triton(x, w, dy, layout, block)
  print((ry - ty).abs().max())
  print((rdx - tdx).abs().max())
  # print(rdx[0,0,0,0], tdx[0,0,0,0])
  #print(rdx)
  #print(tdx)
  #print((ry - ty).abs().max())
  #print((rdx - tdx).abs().max())
  #print((rdw - tdw).abs().max())
  #print(rdx)
  #print(tdx)

#############
# Run tests #
#############

if __name__ == '__main__':
  # test softmax
  #test_softmax(3, 2, 256, 2048, 0.5, 0.7, 16)
  # test matmul
  #for mode in ['sdd', 'dsd', 'dds']:
  #  test_mm(3, 2, 256, 512, 384, 0.5, mode, False, False, 32)
  #  test_mm(3, 2, 256, 512, 384, 0.5, mode, True, False, 32)
  #  test_mm(3, 2, 256, 512, 384, 0.5, mode, False, True, 32)
  #  test_mm(3, 2, 256, 512, 384, 0.5, mode, True, True, 32)
  test_conv2d(8, 16, 16, 16, 16, 3, 3, 0., 16)
  pass