from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.optim import SGD
import extra.waifu2x
from extra.kinne import KinneDir
import sys
import os
import random
import json
import numpy

# amount of context erased by model
CONTEXT = 7

def get_sample_count(samples_dir):
  try:
    samples_dir_count_file = open(samples_dir + "/sample_count.txt", "r")
    v = samples_dir_count_file.readline()
    samples_dir_count_file.close()
    return int(v)
  except:
    return 0

def set_sample_count(samples_dir, sc):
  samples_dir_count_file = open(samples_dir + "/sample_count.txt", "w")
  samples_dir_count_file.write(str(sc) + "\n")
  samples_dir_count_file.close()

if len(sys.argv) < 2:
  print("python3 -m examples.vgg7 import MODELJSON MODELDIR")
  print(" imports a waifu2x JSON vgg_7 model, i.e. waifu2x/models/vgg_7/art/scale2.0x_model.json")
  print(" into a directory of float binaries along with a meta.txt file containing tensor sizes")
  print(" weight tensors are ordered in tinygrad/ncnn form, as so: (outC,inC,H,W)")
  print(" *this format is used by all other commands in this program*")
  print("python3 -m examples.vgg7 execute MODELDIR IMG_IN IMG_OUT")
  print(" given an already-nearest-neighbour-scaled image, runs vgg7 on it")
  print(" output image has 7 pixels removed on all edges")
  print(" do not run on large images, will have *hilarious* RAM use")
  print("python3 -m examples.vgg7 execute_full MODELDIR IMG_IN IMG_OUT")
  print(" does the 'whole thing' (padding, tiling)")
  print(" safe for large images, etc.")
  print("python3 -m examples.vgg7 new MODELDIR")
  print(" creates a new model (experimental)")
  print("python3 -m examples.vgg7 train MODELDIR SAMPLES_DIR ROUNDS ROUNDS_SAVE")
  print(" trains a model (experimental)")
  print(" (how experimental? well, every time I tried it, it flooded w/ NaNs)")
  print(" note: ROUNDS < 0 means 'forever'. ROUNDS_SAVE <= 0 is not a good idea.")
  print(" expects roughly execute's input as SAMPLES_DIR/IDXa.png")
  print(" expects roughly execute's output as SAMPLES_DIR/IDXb.png")
  print(" (i.e. my_samples/0a.png is the first pre-nearest-scaled image,")
  print("       my_samples/0b.png is the first original image)")
  print(" in addition, SAMPLES_DIR/samples_count.txt indicates sample count")
  print(" won't pad or tile, so keep image sizes sane")
  print("python3 -m examples.vgg7 samplify IMG_A IMG_B SAMPLES_DIR SIZE")
  print(" creates overlapping micropatches (SIZExSIZE w/ 7-pixel border) for training")
  print(" maintains/creates samples_count.txt automatically")
  print(" unlike training, IMG_A must be exactly half the size of IMG_B")
  sys.exit(1)

cmd = sys.argv[1]
vgg7 = extra.waifu2x.Vgg7()

def nansbane(p):
  if numpy.isnan(numpy.min(p.data)):
    raise Exception("A NaN in the model has been detected. This model will not be interacted with to prevent further damage.")

def load_and_save(path, save):
  if save:
    for v in vgg7.get_parameters():
      nansbane(v)
  kn = KinneDir(model, save)
  kn.parameters(vgg7.get_parameters())
  kn.close()
  if not save:
    for v in vgg7.get_parameters():
      nansbane(v)

if cmd == "import":
  src = sys.argv[2]
  model = sys.argv[3]

  vgg7.load_waifu2x_json(json.load(open(src, "rb")))

  os.mkdir(model)
  load_and_save(model, True)
elif cmd == "execute":
  model = sys.argv[2]
  in_file = sys.argv[3]
  out_file = sys.argv[4]

  load_and_save(model, False)

  extra.waifu2x.image_save(out_file, vgg7.forward(Tensor(extra.waifu2x.image_load(in_file))).data)
elif cmd == "execute_full":
  model = sys.argv[2]
  in_file = sys.argv[3]
  out_file = sys.argv[4]

  load_and_save(model, False)

  extra.waifu2x.image_save(out_file, vgg7.forward_tiled(extra.waifu2x.image_load(in_file), 156))
elif cmd == "new":
  model = sys.argv[2]

  os.mkdir(model)
  load_and_save(model, True)
elif cmd == "train":
  model = sys.argv[2]
  samples_base = sys.argv[3]
  samples_count = get_sample_count(samples_base)
  rounds = int(sys.argv[4])
  rounds_per_save = int(sys.argv[5])

  load_and_save(model, False)

  print("Training...")
  # Adam has a tendency to destroy the state of the network when restarted
  # Plus it's slower
  optim = SGD(vgg7.get_parameters())

  rnum = 0
  while True:
    # The way the -1 option works is that rnum is never -1.
    if rnum == rounds:
      break

    sample_idx = random.randint(0, samples_count - 1)

    x_img = extra.waifu2x.image_load(samples_base + "/" + str(sample_idx) + "a.png")
    y_img = extra.waifu2x.image_load(samples_base + "/" + str(sample_idx) + "b.png")

    sample_x = Tensor(x_img, requires_grad = False)
    sample_y = Tensor(y_img, requires_grad = False)

    # magic code roughly from readme example
    # An explaination, in case anyone else has to go down this path:
    # This runs the actual network normally
    out = vgg7.forward(sample_x)
    # Subtraction determines error here (as this is an image, not classification).
    # *Abs is the important bit* - at least for me, anyway.
    # The training process seeks to minimize this 'loss' value.
    # Minimization of loss *tends towards negative infinity*, so without the abs,
    #  or without an implicit abs (the mul in the README),
    #  loss will always go haywire in one direction or another.
    # Mean determines how errors are treated.
    # Do not use Sum. I tried that. It worked while I was using 1x1 patches...
    # Then it went exponential.
    # Also, Mean goes *after* abs. I realize this should have been obvious to me.
    loss = sample_y.sub(out).abs().mean()
    # This is the bit where tinygrad works backward from the loss
    optim.zero_grad()
    loss.backward()
    # And this updates the parameters
    optim.step()

    print("Round " + str(rnum) + " : " + str(loss.max().data[0]))

    if (rnum % rounds_per_save) == 0:
      print("Saving")
      load_and_save(model, True)
    rnum = rnum + 1

  # if we were told to save every round, we already saved
  if rounds_per_save != 1:
    print("Done with all rounds, saving")
    load_and_save(model, True)

elif cmd == "samplify":
  a_img = sys.argv[2]
  b_img = sys.argv[3]
  samples_base = sys.argv[4]
  sample_size = int(sys.argv[5])
  samples_count = get_sample_count(samples_base)

  # This bit is interesting because it actually does some work.
  # Not much, but some work.
  a_img = extra.waifu2x.image_load(a_img)
  b_img = extra.waifu2x.image_load(b_img)

  # as with the main library body,
  # Y X order is used here

  # assertion before pre-upscaling is performed
  assert a_img.shape[2] == (b_img.shape[2] // 2)
  assert a_img.shape[3] == (b_img.shape[3] // 2)

  # pre-upscaling - this matches the sizes (and coordinates)
  a_img = a_img.repeat(2, 2).repeat(2, 3)

  samples_added = 0

  # actual patch extraction
  for posy in range(CONTEXT, b_img.shape[2] - (CONTEXT + sample_size - 1), sample_size):
    for posx in range(CONTEXT, b_img.shape[3] - (CONTEXT + sample_size - 1), sample_size):
      # this is a viable patch location, add it
      # note the ranges here:
      #  + there are always CONTEXT pixels *before* the point
      #  + with no subtraction at the end, there'd already be a pixel *at* the point,
      #     as ranges are exclusive
      #  + additionally, there are sample_size - 1 additional sample pixels
      #  + additionally, there are CONTEXT additional pixels
      #  + therefore there are CONTEXT + sample_size pixels *at & after* the point
      patch_x = a_img[:, :, posy - CONTEXT : posy + CONTEXT + sample_size, posx - CONTEXT : posx + CONTEXT + sample_size]
      patch_y = b_img[:, :, posy : posy + sample_size, posx : posx + sample_size]

      extra.waifu2x.image_save(samples_base + "/" + str(samples_count) + "a.png", patch_x)
      extra.waifu2x.image_save(samples_base + "/" + str(samples_count) + "b.png", patch_y)
      samples_count += 1
      samples_added += 1

  print("Added " + str(samples_added) + " samples")
  set_sample_count(samples_base, samples_count)

else:
  print("unknown command")

