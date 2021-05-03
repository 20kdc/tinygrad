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

# ----------------------------------

mul_program = PotentialProgram(["in_av", "in_bv", "out_ov"], [], """
  // iterationID is the index of out_ov to set
  uint inAIdx = iterationID % uint(in_av_len);
  uint inBIdx = iterationID % uint(in_bv_len);
  out_ov[iterationID] = in_av[inAIdx] * in_bv[inBIdx];
""")

class Mul(Function):
  def forward(ctx, x, y):
    # print(ctx.vk_mgr)
    res_shape = tuple(np.max([x.shape, y.shape], 0))
    iterations = 1
    for v in res_shape:
      iterations *= v
    res = GPUBuffer(res_shape)
    mpi = mul_program.instance(ctx.vk_mgr, [x.cl_v, y.cl_v, res.cl_v], iterations)
    ctx.vk_mgr.sequence().record(kp.OpAlgoDispatch(mpi)).eval()
    return res
  def backward():
    raise Exception("NYI")

