import functools
import kp
import numpy as np
from .tensor import Function, register, GPUBuffer, Tensor, Device

global WAVEFRONT
# alternatively, "warp"... if you're a HERETIC! *BLAM*
WAVEFRONT = 64

# --- Simplified Program Runtime ---

class PotentialProgram:
  def __init__(self, tensorids, specids, code):
    header = "#version 450\nlayout (local_size_x = " + str(WAVEFRONT) + ") in;\n"
    binding_id = 0
    for v in tensorids:
      attr = None
      if v.startswith("in_"):
        attr = "readonly"
      elif v.startswith("out_"):
        attr = "writeonly"
      else:
        raise Exception("Unknown tensor mode")
      header += "layout (set = 0, binding = " + str(binding_id) + ") buffer " + v + "_buf { " + attr + " restrict float " + v + "[]; };\n"
      header += "layout (constant_id = " + str(binding_id) + ") const float " + v + "_len = 0;\n"
      binding_id += 1

    header += "layout (constant_id = " + str(binding_id) + ") const float iterations_l = 0;\n"
    binding_id += 1

    header += "layout (constant_id = " + str(binding_id) + ") const float iterations_h = 0;\n"
    binding_id += 1

    for v in specids:
      header += "layout (constant_id = " + str(binding_id) + ") const float " + v + " = 0;\n"
      binding_id += 1

    header += "void main() {\n"
    header += "uint iterationID = gl_GlobalInvocationID.x + (gl_GlobalInvocationID.y * 65536);\n"
    header += "if (iterationID >= (uint(iterations_l) + (uint(iterations_h) * 65536))) return;\n"
    footer = "\n}"

    self.spirv = kp.Shader.compile_source(header + code + footer)

  def instance(self, mgr, native_tensors, iterations, regularconsts = [], pushconsts = []):
    specconsts = []
    for v in native_tensors:
      specconsts.append(v.size())
    specconsts.append(iterations & 0xFFFF) # iterations_l
    specconsts.append(iterations >> 16) # iterations_h
    specconsts += regularconsts
    # this must occur after the above iterations constants
    iterations = (iterations + (WAVEFRONT - 1)) // WAVEFRONT
    return mgr.algorithm(
        native_tensors,
        self.spirv,
        (iterations & 0xFFFF, iterations >> 16, 1),
        specconsts,
        pushconsts
    )

def gpu_upload_buffers(kpm, buffers):
  last_seq = kpm.sequence()
  last_seq.eval_async(kp.OpTensorSyncDevice(buffers))
  last_seq.eval_await()

def gpu_download_buffers(kpm, buffers):
  last_seq = kpm.sequence()
  last_seq.eval_async(kp.OpTensorSyncLocal(buffers))
  last_seq.eval_await()

def shape_len(shape):
  iterations = 1
  for v in shape:
    iterations *= v
  return iterations

# ----------------------------------

def basic_2p(ctx, x, y, program):
  # print(ctx.vk_mgr)
  res_shape = tuple(np.max([x.shape, y.shape], 0))
  iterations = shape_len(res_shape)
  res = GPUBuffer(res_shape)
  mpi = mul_program.instance(ctx.vk_mgr, [x.cl_v, y.cl_v, res.cl_v], iterations)
  ctx.vk_mgr.sequence().record(kp.OpAlgoDispatch(mpi)).eval()
  return res

# ----------------------------------

mul_program = PotentialProgram(["in_av", "in_bv", "out_ov"], [], """
  // iterationID is the index of out_ov to set
  uint inAIdx = iterationID % uint(in_av_len);
  uint inBIdx = iterationID % uint(in_bv_len);
  out_ov[iterationID] = in_av[inAIdx] * in_bv[inBIdx];
""")

class Mul(Function):
  def forward(ctx, x, y):
    return basic_2p(ctx, x, y, mul_program)
  def backward():
    raise Exception("NYI")

# ----------------------------------

add_program = PotentialProgram(["in_av", "in_bv", "out_ov"], [], """
  // iterationID is the index of out_ov to set
  uint inAIdx = iterationID % uint(in_av_len);
  uint inBIdx = iterationID % uint(in_bv_len);
  out_ov[iterationID] = in_av[inAIdx] + in_bv[inBIdx];
""")

class Add(Function):
  def forward(ctx, x, y):
    return basic_2p(ctx, x, y, add_program)
  def backward():
    raise Exception("NYI")

# ----------------------------------

sub_program = PotentialProgram(["in_av", "in_bv", "out_ov"], [], """
  // iterationID is the index of out_ov to set
  uint inAIdx = iterationID % uint(in_av_len);
  uint inBIdx = iterationID % uint(in_bv_len);
  out_ov[iterationID] = in_av[inAIdx] - in_bv[inBIdx];
""")

class Sub(Function):
  def forward(ctx, x, y):
    return basic_2p(ctx, x, y, sub_program)
  def backward():
    raise Exception("NYI")

# ----------------------------------

class Reshape(Function):
  def forward(ctx, x, shape):
    original_len = shape_len(x.shape)
    # we need to determine 'true shape' (i.e. accounting for that pesky -1)
    true_shape = list(shape)
    main_mul = 1
    for v in true_shape:
      if v != -1:
        main_mul *= v
    for i in range(len(true_shape)):
      if true_shape[i] == -1:
        true_shape[i] = original_len // main_mul
    res = GPUBuffer(true_shape)
    ctx.vk_mgr.sequence().record(kp.OpTensorCopy([x.cl_v, res.cl_v])).eval()
    return res
  def backward():
    raise Exception("NYI")

# ----------------------------------

# TODO TODO THIS IS CLEARLY WRONG TODO TODO

conv2d_program = PotentialProgram(["in_av", "in_bv", "out_ov"], [], """
  // iterationID is the index of out_ov to set
  uint inAIdx = iterationID % uint(in_av_len);
  uint inBIdx = iterationID % uint(in_bv_len);
  out_ov[iterationID] = 1.0; //in_av[inAIdx] + in_bv[inBIdx];
""")

class Conv2D(Function):
  def forward(ctx, x, y):
    # print(ctx.vk_mgr)
    res_shape = list(x.shape)
    assert len(res_shape) == 4
    res_shape[1] = y.shape[0]
    res_shape[2] -= 2
    res_shape[3] -= 2
    iterations = shape_len(res_shape)
    res = GPUBuffer(res_shape)
    mpi = conv2d_program.instance(ctx.vk_mgr, [x.cl_v, y.cl_v, res.cl_v], iterations)
    ctx.vk_mgr.sequence().record(kp.OpAlgoDispatch(mpi)).eval()
    return res
  def backward():
    raise Exception("NYI")

# ----------------------------------

relu_program = PotentialProgram(["in_av", "out_ov"], [], """
  out_ov[iterationID] = max(in_av[iterationID], 0.0);
""")

class ReLU(Function):
  def forward(ctx, x):
    iterations = shape_len(x.shape)
    res = GPUBuffer(x.shape)
    mpi = add_program.instance(ctx.vk_mgr, [x.cl_v, res.cl_v], iterations)
    ctx.vk_mgr.sequence().record(kp.OpAlgoDispatch(mpi)).eval()
    return res
  def backward():
    raise Exception("NYI")

