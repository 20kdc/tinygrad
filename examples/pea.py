from tinygrad.tensor import Tensor
from tinygrad.optim import SGD
import extra.pea
from extra.kinne import KinneDir
import sys
import os
import random
import json
import numpy

if len(sys.argv) < 3:
  print("MDL is model config - can be 0 or 1")
  print("python3 -m examples.pea MDL execute RAW_IN RAW_OUT")
  print(" given raw float32 stereo audio, runs PEA on it")
  print(" output has 7 samples removed on both edges")
  print(" do not run on large audio, will have *hilarious* RAM use")
  print("python3 -m examples.pea MDL execute_full RAW_IN RAW_OUT")
  print(" like execute but scales and doesn't crop the edges")
  print("python3 -m examples.pea MDL new")
  print(" reinitializes model")
  print("python3 -m examples.pea MDL train ROUNDS ROUNDS_SAVE")
  print(" trains a model")
  print(" note: ROUNDS < 0 means 'forever'. ROUNDS_SAVE <= 0 is not a good idea.")
  print(" expects roughly execute's input as pea/dataset/i.raw")
  print(" expects roughly execute's output as pea/dataset/o.raw")
  print(" automatically selects only part of the data, so don't worry about size")
  sys.exit(1)

cmd = sys.argv[2]
pea = extra.pea.Pea(extra.pea.PEA_CONFIGS[sys.argv[1]])

MODEL = "pea/model." + sys.argv[1]

def nansbane(p):
  if numpy.isnan(numpy.min(p.data)):
    raise Exception("A NaN in the model has been detected. This model will not be interacted with to prevent further damage.")

def load_and_save(save):
  if save:
    for v in pea.get_parameters():
      nansbane(v)
  kn = KinneDir(MODEL, save)
  kn.parameters(pea.get_parameters())
  kn.close()
  if not save:
    for v in pea.get_parameters():
      nansbane(v)

if cmd == "execute":
  in_file = sys.argv[3]
  out_file = sys.argv[4]

  load_and_save(False)

  nd = numpy.fromfile(in_file, "<f4").reshape((-1, 2))
  nd = extra.pea.into_cdat_i(nd)
  nd = pea.forward(Tensor(nd)).data
  nd = extra.pea.from_cdat_o(nd)
  nd.astype("<f4", "C").tofile(out_file)
elif cmd == "execute_full":
  in_file = sys.argv[3]
  out_file = sys.argv[4]

  load_and_save(False)

  in_file = numpy.fromfile(in_file, "<f4").reshape((-1, 2))
  pea.forward_tiled(in_file, 65536).astype("<f4", "C").tofile(out_file)
elif cmd == "new":
  os.mkdir(MODEL)
  load_and_save(True)
elif cmd == "train":
  rounds = int(sys.argv[3])
  rounds_per_save = int(sys.argv[4])

  window_i_size = 64
  # Context is N *input* samples, therefore N2 *output* samples,
  #  therefore Context2 is N4 *output* samples
  window_o_offs = pea.context() * 2
  window_o_size = (window_i_size * 2) - (window_o_offs * 2)

  dataset_i = numpy.memmap("pea/dataset/i.raw", dtype="<f4").reshape((-1, 2))
  dataset_o = numpy.memmap("pea/dataset/o.raw", dtype="<f4").reshape((-1, 2))
  dataset_i_samples = dataset_i.shape[0]

  load_and_save(False)

  print("Training...")
  # Adam has a tendency to destroy the state of the network when restarted
  # Plus it's slower
  optim = SGD(pea.get_parameters())

  rnum = 0
  while True:
    # The way the -1 option works is that rnum is never -1.
    if rnum == rounds:
      break

	# Select & pull sample
    sample_i_start = random.randint(0, dataset_i_samples - window_i_size)
    sample_i_slice = dataset_i[sample_i_start : sample_i_start + window_i_size]

    sample_o_start = (sample_i_start * 2) + window_o_offs
    sample_o_slice = dataset_o[sample_o_start : sample_o_start + window_o_size]

	# Prepare sample
    sample_x = Tensor(extra.pea.into_cdat_i(sample_i_slice), requires_grad = False)
    sample_y = Tensor(extra.pea.into_cdat_o(sample_o_slice), requires_grad = False)

    # magic code roughly from readme example
    # An explaination, in case anyone else has to go down this path:
    # This runs the actual network normally
    out = pea.forward(sample_x)
    loss = pea.loss(out, sample_y)
    # This is the bit where tinygrad works backward from the loss
    optim.zero_grad()
    loss.backward()
    # And this updates the parameters
    optim.step()

    # warning: used by sample probability adjuster
    loss_indicator = loss.max().data[0]
    print("Round " + str(rnum) + " : " + str(loss_indicator))

    if (rnum % rounds_per_save) == 0:
      print("Saving")
      load_and_save(True)

    # Update round state
    # Number
    rnum = rnum + 1

  # if we were told to save every round, we already saved
  if rounds_per_save != 1:
    print("Done with all rounds, saving")
    load_and_save(True)

else:
  print("unknown command")

