#from builtins import bytes
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


tmp_ = torch.rand(1,1).cuda()

SRU_CODE = """
extern "C" {

    __forceinline__ __device__ float sigmoidf(float x)
    {
        return 1.f / (1.f + expf(-x));
    }

    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }


__global__ void sru_fwd(const float * __restrict__ u,
  const float * __restrict__ x,
    const float * __restrict__ b,
      const float * __restrict__ init_c,
        const float * __restrict__ mask_h,
          const int seq_len,
            const int n_batch,
              const int d_out,
                const int k,
                  float * __restrict__ h, float * __restrict__ c,
                  const int use_tanh) {
  /*
   * u (seq_len, n_batch, d_out, k)
   * x (seq_len, n_batch, d_in) | NULL
   * b (2, d_out)
   * init_c (n_batch, d_out)
   * mask_h (n_batch, d_out)
   *
   * h (seq_len, n_batch, d_out)
   * c (seq_len, n_batch, d_out)
   *
   * */

  // k==3 indicates x[-1] == d_out, otherwise k=4 x=NULL
  assert ((k == 3) || (x == NULL));

  int ncols = n_batch * d_out;
  int ncols_u = n_batch * d_out * k;
  // if k==4, x is part of u
  int ncols_x = (k == 3) ? ncols : ncols_u;

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= ncols)
    return;

  const float bf = * (b + (col % d_out));
  const float br = * (b + (col % d_out) + d_out);
  const float mask = (mask_h == 0) ? 1.0 : * (mask_h + col);

  // timestep 0
  const float * Wcx_p = u + col * k;
  const float * Wfx_p = u + col * k + 1;
  const float * Wrx_p = u + col * k + 2;
  const float * x_p = (k == 3) ? x + col : u + col * k + 3;
  const float * prev_c_p = init_c + col;

  float * c_p = c + col;
  float * h_p = h + col;

  for (int i = 0; i < seq_len; i++) {
    float inner_c = * Wcx_p;
    float fg = sigmoidf(( * Wfx_p) + bf);
    float rg = sigmoidf(( * Wrx_p) + br);

    * c_p = fg * ( * prev_c_p) + (1 - fg) * (inner_c); * h_p = rg * (mask * tanh( * c_p)) + (1 - rg) * ( * x_p);
    // move to next point
    Wcx_p += ncols_u;
    Wfx_p += ncols_u;
    Wrx_p += ncols_u;
    x_p += ncols_x;
    prev_c_p = c_p;
    c_p += ncols;
    h_p += ncols;
  }

}

__global__ void sru_bwd(const float * __restrict__ u,
  const float * __restrict__ x,
    const float * __restrict__ b,
      const float * __restrict__ init_c,
        const float * __restrict__ mask_h,
          const float * __restrict__ c,
            const float * __restrict__ grad_h,
              const float * __restrict__ grad_last_c,
                const int seq_len,
                  const int n_batch,
                    const int d_out,
                      const int k,
                        float * __restrict__ grad_u, float * __restrict__ grad_x,
                        float * __restrict__ grad_b, float * __restrict__ grad_init,
                        int use_tanh) {
  /*
   * u (seq_len, n_batch, d_out, k)
   * x (seq_len, n_batch, d_in) | NULL
   * b (2, d_out)
   * init_c (n_batch, d_out)
   * mask_h (n_batch, d_out)
   * c (seq_len, n_batch, d_out)
   * grad_h (seq_len, n_batch, d_out)
   * grad_last_c (n_batch, d_out)
   *
   * grad_u (seq_len, n_batch, d_out, k)
   * grad_x (seq_len, n_batch, d_in) | NULL
   * grad_b (2, batch, d_out)
   * grad_init (n_batch, d_out)
   *
   *
   * According to backprop, at every time step and every position we have
   * grad_c = grad_h*rg*grad_g(mask*c)*mask;
   * grad_x = grad_h*(1-rg);
   * grad_rg = grad_h*(g(mask*c)-x)             from (5)
   *
   * grad_inner_c = grad_c*(1-fg);
   * grad_fg = grad_c*(prev_c-inner_c)
   * grad_prev_c = grad_c*fg;                   from (4)
   *
   * grad_Wrx = grad_br = grad_rg*rg*(1-rg);    from (3)
   *
   * grad_Wfx = grad_bf = grad_fg*fg*(1-fg);    from (2)
   *
   * grad_Wcx = grad_inner_c                    from (1)
   * */

  assert((k == 3) || (x == NULL));
  assert((k == 3) || (grad_x == NULL));

  int ncols = n_batch * d_out;
  int ncols_u = n_batch * d_out * k;
  int ncols_x = (k == 3) ? ncols : ncols_u;

  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= ncols)
    return;

  const float bf = * (b + (col % d_out));
  const float br = * (b + (col % d_out) + d_out);
  const float mask = (mask_h == 0) ? 1.0 : * (mask_h + col);

  float * grad_bf_p = grad_b + col;
  float * grad_br_p = grad_b + col + ncols;
  float * grad_c_p = grad_init + col;

  // start from timestep T
  const float * Wcx_p = u + (seq_len - 1) * ncols_u + col * k;
  const float * Wfx_p = u + (seq_len - 1) * ncols_u + col * k + 1;
  const float * Wrx_p = u + (seq_len - 1) * ncols_u + col * k + 2;
  const float * x_p = (k == 3) ? x + (seq_len - 1) * ncols + col : u + (seq_len - 1) * ncols_u + col * k + 3;
  const float * c_p = c + (seq_len - 1) * ncols + col;

  const float * grad_last_c_p = grad_last_c + col;

  const float * grad_h_p = grad_h + (seq_len - 1) * ncols + col;

  float * grad_Wcx_p = grad_u + (seq_len - 1) * ncols_u + col * k;
  float * grad_Wfx_p = grad_u + (seq_len - 1) * ncols_u + col * k + 1;
  float * grad_Wrx_p = grad_u + (seq_len - 1) * ncols_u + col * k + 2;
  float * grad_x_p = (k == 3) ? grad_x + (seq_len - 1) * ncols + col : grad_u + (seq_len - 1) * ncols_u + col * k + 3;

  * grad_br_p = 0; * grad_bf_p = 0;
  for (int i = seq_len - 1; i >= 0; i--) {
    const float fg = sigmoidf( * Wfx_p + bf);
    const float rg = sigmoidf( * Wrx_p + br);
    // grad_last_c is the c at time step i
    * grad_c_p = * grad_last_c_p;
    // grad_c = grad_h*rg*grad_g(mask*c)*mask
    * grad_c_p += ( * grad_h_p) * rg * (1 - tanh(mask * ( * c_p)) * tanh(mask * ( * c_p))) * mask;
    // grad_x = grad_h*(1-rg)
    * grad_x_p = ( * grad_h_p) * (1 - rg);
    // grad_rg = grad_h*(g(mask*c)-x)
    const float grad_rg = ( * grad_h_p) * (tanh(mask * ( * c_p)) - ( * x_p));
    // grad_inner_c = grad_c*(1-fg)
    * grad_Wcx_p = ( * grad_c_p) * (1 - fg);
    // grad_fg = grad_c*(prev_c-inner_c)
    const float prev_c = i == 0 ? * (init_c + col) : * (c_p - ncols);
    const float grad_fg = ( * grad_c_p) * (prev_c - ( * Wcx_p));
    // grad_prev_c = grad_c*fg
    * grad_c_p = ( * grad_c_p) * fg;
    grad_last_c_p = grad_c_p;
    // grad_Wrx = grad_br = grad_rg*rg*(1-rg)
    * grad_Wrx_p = grad_rg * rg * (1 - rg); * grad_br_p += * grad_Wrx_p;
    // grad_Wfx = grad_bf = grad_fg*fg*(1-fg)
    * grad_Wfx_p = grad_fg * fg * (1 - fg); * grad_bf_p += * grad_Wfx_p;
    // move to next point
    Wcx_p -= ncols_u;
    Wfx_p -= ncols_u;
    Wrx_p -= ncols_u;
    x_p -= ncols_x;
    c_p -= ncols;
    grad_h_p -= ncols;
    grad_Wcx_p -= ncols_u;
    grad_Wfx_p -= ncols_u;
    grad_Wrx_p -= ncols_u;
    grad_x_p -= ncols_x;

  }

}

__global__ void sru_bi_fwd(const float * __restrict__ u,
  const float * __restrict__ x,
    const float * __restrict__ b,
      const float * __restrict__ init_c,
        const float * __restrict__ mask_h,
          const int seq_len,
            const int n_batch,
              const int d_out,
                const int k,
                  float * __restrict__ h, float * __restrict__ c,
                  const int use_tanh) {
  /*
   * u (seq_len, n_batch, 2, d_out, k)
   * x (seq_len, n_batch, d_in) | NULL
   * b (2, 2, d_out)
   * init_c (n_batch, 2, d_out)
   * mask_h (n_batch, 2, d_out)
   *
   * h (seq_len, n_batch, 2, d_out)
   * c (seq_len, n_batch, 2, d_out)
   *
   * */

  assert ((k == 3) || (x == NULL));
  assert ((k == 3) || (k == 4));

  int ncols = n_batch * 2 * d_out;
  int ncols_u = ncols * k;
  // if k==4, x is part of u
  int ncols_x = (k == 3) ? ncols : ncols_u;

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= ncols)
    return;

  const int d_2_out = 2 * d_out;
  const float bf = * (b + (col % d_2_out));
  const float br = * (b + (col % d_2_out) + d_2_out);
  const float mask = (mask_h == 0) ? 1.0 : * (mask_h + col);
  // forward encoding if flip==0
  const bool flip = (col % d_2_out) >= d_out;
  // timestep 0
  const float * Wcx_p = u + col * k;
  const float * Wfx_p = u + col * k + 1;
  const float * Wrx_p = u + col * k + 2;
  const float * x_p = (k == 3) ? x + col : u + col * k + 3;
  const float * prev_c_p = init_c + col;

  float * c_p = c + col;
  float * h_p = h + col;
  // backward encoding should start from timestep T
  if (flip) {
    Wcx_p += (seq_len - 1) * ncols_u;
    Wfx_p += (seq_len - 1) * ncols_u;
    Wrx_p += (seq_len - 1) * ncols_u;
    x_p += (seq_len - 1) * ncols_x;
    c_p += (seq_len - 1) * ncols;
    h_p += (seq_len - 1) * ncols;
  }

  //set proper incremental direction
  int ncols_u_ = flip ? -ncols_u : ncols_u;
  int ncols_x_ = flip ? -ncols_x : ncols_x;
  int ncols_ = flip ? -ncols : ncols;

  for (int i = 0; i < seq_len; i++) {
    float inner_c = * Wcx_p;
    float fg = sigmoidf(( * Wfx_p) + bf);
    float rg = sigmoidf(( * Wrx_p) + br);

    * c_p = fg * ( * prev_c_p) + (1 - fg) * (inner_c); * h_p = rg * (mask * tanh( * c_p)) + (1 - rg) * ( * x_p);
    // move to next point
    Wcx_p += ncols_u_;
    Wfx_p += ncols_u_;
    Wrx_p += ncols_u_;
    x_p += ncols_x_;
    prev_c_p = c_p;
    c_p += ncols_;
    h_p += ncols_;
  }

}

__global__ void sru_bi_bwd(const float * __restrict__ u,
  const float * __restrict__ x,
    const float * __restrict__ b,
      const float * __restrict__ init_c,
        const float * __restrict__ mask_h,
          const float * __restrict__ c,
            const float * __restrict__ grad_h,
              const float * __restrict__ grad_last_c,
                const int seq_len,
                  const int n_batch,
                    const int d_out,
                      const int k,
                        float * __restrict__ grad_u, float * __restrict__ grad_x,
                        float * __restrict__ grad_b, float * __restrict__ grad_init,
                        int use_tanh) {
  /*
   * u (seq_len, n_batch, directions, d_out, k)
   * x (seq_len, n_batch, d_in) | NULL
   * b (2, directions, d_out)
   * init_c (n_batch, directions, d_out)
   * mask_h (n_batch, directions, d_out)
   * c (seq_len, n_batch, directions, d_out)
   * grad_h (seq_len, n_batch, directions, d_out)
   * grad_last_c (n_batch, directions, d_out)
   *
   * grad_u (seq_len, n_batch, directions, d_out, k)
   * grad_x (seq_len, n_batch, d_in) | NULL
   * grad_b (2, batch, directions, d_out)
   * grad_init (n_batch, directions, d_out)
   * */

  assert((k == 3) || (x == NULL));
  assert((k == 3) || (grad_x == NULL));
  assert((k == 3) || (k == 4));

  int ncols = n_batch * 2 * d_out;
  int ncols_u = ncols * k;
  int ncols_x = (k == 3) ? ncols : ncols_u;
  int d_2_out = 2 * d_out;

  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= ncols)
    return;

  const float bf = * (b + (col % d_2_out));
  const float br = * (b + (col % d_2_out) + d_2_out);
  const float mask = (mask_h == 0) ? 1.0 : * (mask_h + col);
  // forward encoding if flip==0
  const bool flip = (col % d_2_out) >= d_out;

  float * grad_bf_p = grad_b + col;
  float * grad_br_p = grad_b + col + ncols;
  float * grad_c_p = grad_init + col;

  // forward encoding do BP from timestep T
  const float * Wcx_p = u + (seq_len - 1) * ncols_u + col * k;
  const float * Wfx_p = u + (seq_len - 1) * ncols_u + col * k + 1;
  const float * Wrx_p = u + (seq_len - 1) * ncols_u + col * k + 2;
  const float * x_p =
    (k == 3) ?
    x + (seq_len - 1) * ncols + col :
    u + (seq_len - 1) * ncols_u + col * k + 3;
  const float * c_p = c + (seq_len - 1) * ncols + col;

  const float * grad_last_c_p = grad_last_c + col;

  const float * grad_h_p = grad_h + (seq_len - 1) * ncols + col;

  float * grad_Wcx_p = grad_u + (seq_len - 1) * ncols_u + col * k;
  float * grad_Wfx_p = grad_u + (seq_len - 1) * ncols_u + col * k + 1;
  float * grad_Wrx_p = grad_u + (seq_len - 1) * ncols_u + col * k + 2;
  float * grad_x_p =
    (k == 3) ?
    grad_x + (seq_len - 1) * ncols + col :
    grad_u + (seq_len - 1) * ncols_u + col * k + 3;

  // BP from timestep 0 if flip==1
  if (flip) {
    Wcx_p -= (seq_len - 1) * ncols_u;
    Wfx_p -= (seq_len - 1) * ncols_u;
    Wrx_p -= (seq_len - 1) * ncols_u;
    x_p -= (seq_len - 1) * ncols_x;
    c_p -= (seq_len - 1) * ncols;

    grad_Wcx_p -= (seq_len - 1) * ncols_u;
    grad_Wfx_p -= (seq_len - 1) * ncols_u;
    grad_Wrx_p -= (seq_len - 1) * ncols_u;
    grad_x_p -= (seq_len - 1) * ncols_x;
    grad_h_p -= (seq_len - 1) * ncols;
  }

  // set proper incremental direction
  int ncols_u_ = flip ? -ncols_u : ncols_u;
  int ncols_x_ = flip ? -ncols_x : ncols_x;
  int ncols_ = flip ? -ncols : ncols;

  // init br, bf 
  * grad_br_p = 0; * grad_bf_p = 0;
  for (int i = seq_len - 1; i >= 0; i--) {
    const float fg = sigmoidf( * Wfx_p + bf);
    const float rg = sigmoidf( * Wrx_p + br);
    // grad_last_c is the c at time step i
    * grad_c_p = * grad_last_c_p;
    // grad_c = grad_h*rg*grad_g(mask*c)*mask
    * grad_c_p += ( * grad_h_p) * rg * (1 - tanh(mask * ( * c_p)) * tanh(mask * ( * c_p))) * mask;
    // grad_x = grad_h*(1-rg)
    * grad_x_p = ( * grad_h_p) * (1 - rg);
    // grad_rg = grad_h*(g(mask*c)-x)
    const double grad_rg = ( * grad_h_p) * (tanh(mask * ( * c_p)) - ( * x_p));
    // grad_inner_c = grad_c*(1-fg)
    * grad_Wcx_p = ( * grad_c_p) * (1 - fg);
    // grad_fg = grad_c*(prev_c-inner_c)
    const double prev_c = i == 0 ? * (init_c + col) : * (c_p - ncols_);
    const double grad_fg = ( * grad_c_p) * (prev_c - ( * Wcx_p));
    // grad_prev_c = grad_c*fg
    * grad_c_p = ( * grad_c_p) * fg;
    grad_last_c_p = grad_c_p;
    // grad_Wrx = grad_br = grad_rg*rg*(1-rg)
    * grad_Wrx_p = grad_rg * rg * (1 - rg); * grad_br_p += * grad_Wrx_p;
    // grad_Wfx = grad_bf = grad_fg*fg*(1-fg)
    * grad_Wfx_p = grad_fg * fg * (1 - fg); * grad_bf_p += * grad_Wfx_p;

    // move to next point
    Wcx_p -= ncols_u_;
    Wfx_p -= ncols_u_;
    Wrx_p -= ncols_u_;
    x_p -= ncols_x_;
    c_p -= ncols_;
    grad_h_p -= ncols_;
    grad_Wcx_p -= ncols_u_;
    grad_Wfx_p -= ncols_u_;
    grad_Wrx_p -= ncols_u_;
    grad_x_p -= ncols_x_;

  }

}


}
"""

SRU_PROG = Program(SRU_CODE.encode('utf-8'), 'sru_prog.cu'.encode('utf-8'))
SRU_PTX = SRU_PROG.compile()
SRU_MOD = function.Module()
SRU_MOD.load(bytes(SRU_PTX.encode()))
SRU_FWD_FUNC = SRU_MOD.get_function('sru_fwd')
SRU_BWD_FUNC = SRU_MOD.get_function('sru_bwd')
SRU_BiFWD_FUNC = SRU_MOD.get_function('sru_bi_fwd')
SRU_BiBWD_FUNC = SRU_MOD.get_function('sru_bi_bwd')

Stream = namedtuple('Stream', ['ptr'])
SRU_STREAM = Stream(ptr=torch.cuda.current_stream().cuda_stream)

class SRU_Compute(Function):

    def __init__(self, activation_type, d_out, bidirectional=False):
        super(SRU_Compute, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional

    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)/thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d*bidir) if x.dim() == 3 else (batch, d*bidir)
        c = x.new(*size)
        h = x.new(*size)

        FUNC = SRU_FWD_FUNC if not self.bidirectional else SRU_BiFWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            length,
            batch,
            d,
            k_,
            h.data_ptr(),
            c.data_ptr(),
            self.activation_type],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=SRU_STREAM
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.cat((c[-1,:,:d], c[0,:,d:]), dim=1)
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)/thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d*bidir)
        grad_init = x.new(batch, d*bidir)

        # For DEBUG
        #size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        #grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        FUNC = SRU_BWD_FUNC if not self.bidirectional else SRU_BiBWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            c.data_ptr(),
            grad_h.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            length,
            batch,
            d,
            k_,
            grad_u.data_ptr(),
            grad_x.data_ptr() if k_ == 3 else 0,
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.activation_type],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=SRU_STREAM
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=1, use_relu=0):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_type = 2 if use_relu else (1 if use_tanh else 0)

        out_size = n_out*2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        self.size_per_dir = n_out*k
        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir*2 if bidirectional else self.size_per_dir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out*4 if bidirectional else n_out*2
        ))
        self.init_weight()

    def init_weight(self):
        val_range = (3.0/self.n_in)**0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def set_bias(self, bias_val=0):
        n_out = self.n_out
        if self.bidirectional:
            self.bias.data[n_out*2:].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2
            ).zero_())

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight)

        if self.training and (self.dropout>0):
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_((batch, n_out*bidir), self.dropout)
            h, c = SRU_Compute(self.activation_type, n_out, self.bidirectional)(u, input, self.bias, c0, mask_h)
        else:
            h, c = SRU_Compute(self.activation_type, n_out, self.bidirectional)(u, input, self.bias, c0)

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=1, use_relu=0):
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size*2 if bidirectional else hidden_size

        for i in range(num_layers):
            l = SRUCell(
                n_in = self.n_in if i==0 else self.out_size,
                n_out = self.n_out,
                dropout = dropout if i+1 != num_layers else 0,
                rnn_dropout = rnn_dropout,
                bidirectional = bidirectional,
                use_tanh = use_tanh,
                use_relu = use_relu,
            )
            self.rnn_lst.append(l)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3 # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out*dir_
            ).zero_())
            c0 = [ zeros for i in range(self.depth) ]
        else:
            assert c0.dim() == 3    # (depth, batch, n_out*dir_)
            c0 = [ x.squeeze(0) for x in c0.chunk(self.depth, 0) ]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx


