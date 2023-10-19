import sys
import random
import json
import numpy
from pathlib import Path
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import LAMB
from examples.vgg7_helpers.waifu2x import image_load, image_save
from tinygrad.jit import TinyJit

random.seed()

if len(sys.argv) < 2:
  print("python3 -m examples.ienet new MODEL WIDTH DEPTH")
  print("python3 -m examples.ienet execute MODEL WIDTH HEIGHT IMAGE")
  print("python3 -m examples.ienet train MODEL IMAGE ROUNDS ROUNDS_SAVE")
  sys.exit(1)

cmd = sys.argv[1]

parameters = []

def model_forward(x):
  first = True
  for pi in range(0, len(parameters), 2):
    if not first:
      x = x.leakyrelu(0.1)
    x = x.matmul(parameters[pi]) + parameters[pi + 1]
    first = False
  return x

def nansbane(p):
  if numpy.isnan(numpy.min(p.numpy())):
    raise Exception("A NaN in the model has been detected. This model will not be interacted with to prevent further damage.")

def gen_patch(img_w, img_h, patch_x, patch_y, patch_w, patch_h):
  all_initial_data = []
  for y in range(patch_h):
    for x in range(patch_w):
      all_initial_data.append((x + patch_x) / img_w)
      all_initial_data.append((y + patch_y) / img_h)
      all_initial_data.append(1)
  all_initial_data = Tensor(all_initial_data).reshape(patch_w, patch_h, 3)
  all_initial_data.gpu()
  return all_initial_data.realize()

def load_and_save(path, save):
  if save:
    with open(path + ".npy", "wb") as f:
      for v in parameters:
        nansbane(v)
        numpy.save(f, v.numpy())
  else:
    with open(path + ".npy", "rb") as f:
      parameters.clear()
      while True:
        try:
          np = numpy.load(f)
        except EOFError:
          break
        v = Tensor(np)
        parameters.append(v)
        nansbane(v)

if cmd == "new":
  model = sys.argv[2]
  model_width = int(sys.argv[3])
  model_depth = int(sys.argv[4])

  parameters = []
  parameters.append(Tensor.uniform(3, model_width) * 0.25)
  parameters.append(Tensor.uniform(model_width) * 0.25)
  for idx in range(model_depth):
    parameters.append(Tensor.uniform(model_width, model_width) * 0.25)
    parameters.append(Tensor.uniform(model_width) * 0.25)
  parameters.append(Tensor.uniform(model_width, 3) * 0.25)
  parameters.append(Tensor.uniform(3) * 0.25)

  load_and_save(model, True)
elif cmd == "execute":
  model = sys.argv[2]
  img_w = int(sys.argv[3])
  img_h = int(sys.argv[4])
  img = sys.argv[5]
  load_and_save(model, False)
  patch = gen_patch(img_w, img_h, 0, 0, img_w, img_h)
  tensor = model_forward(patch).numpy()
  # (X, Y, 3) -> (1, 3, X, Y)
  tensor = tensor.transpose(2, 0, 1).reshape(1, 3, img_w, img_h)
  image_save(img, tensor)
elif cmd == "train":
  model = sys.argv[2]
  img = sys.argv[3]
  rounds = int(sys.argv[4])
  rounds_per_save = int(sys.argv[5])

  # Load model & image
  load_and_save(model, False)
  img_data = image_load(img)
  img_w = img_data.shape[2]
  img_h = img_data.shape[3]
  # (1, 3, X, Y) -> (X, Y, 3)
  img_data = Tensor(img_data.reshape(3, img_w, img_h).transpose(1, 2, 0), requires_grad = False)

  print("Training...")
  optim = LAMB(parameters)

  PATCH_SIZE = 32

  rnum = 0
  while True:
    # The way the -1 option works is that rnum is never -1.
    if rnum == rounds:
      break

    patch_x = random.randint(0, img_w - PATCH_SIZE)
    patch_y = random.randint(0, img_h - PATCH_SIZE)
    sample_x = gen_patch(img_w, img_h, patch_x, patch_y, PATCH_SIZE, PATCH_SIZE)
    sample_y = img_data[patch_x:patch_x + PATCH_SIZE, patch_y:patch_y + PATCH_SIZE, :]

    out = model_forward(sample_x)
    loss = sample_y.sub(out).abs().transpose(2, 0).mean()
    # This is the bit where tinygrad works backward from the loss
    optim.zero_grad()
    loss.backward()
    # And this updates the parameters
    optim.step()

    loss_indicator = loss.numpy()
    print("Round " + str(rnum) + " : " + str(loss_indicator))

    if (rnum % rounds_per_save) == 0:
      print("Saving")
      load_and_save(model, True)

    # Update round state
    # Number
    rnum = rnum + 1

  # if we were told to save every round, we already saved
  if rounds_per_save != 1:
    print("Done with all rounds, saving")
    load_and_save(model, True)
else:
  print("unknown command")
