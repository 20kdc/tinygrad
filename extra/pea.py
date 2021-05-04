# A custom pet project.

import numpy
from tinygrad.tensor import Tensor

# File Formats

# tinygrad convolution tensor input layout is (1,c,y,x) - and therefore the form for all data used in the project
# tinygrad convolution tensor weight layout is (outC,inC,H,W)

# The Model

class Conv3x1Biased:
  """
  A 3x1 convolution layer with some utility functions.
  """
  def __init__(self, inC, outC, last = False):
    # Massively overstate the weights to get them to be focused on,
    #  since otherwise the biases overrule everything
    self.weight = Tensor.uniform(outC, inC, 1, 3) * 16.0
    # Layout-wise, blatant cheat, but serious_mnist does it. I'd guess channels either have to have a size of 1 or whatever the target is?
    # Values-wise, entirely different blatant cheat.
    # In most cases, use uniform bias, but tiny.
    # For the last layer, use just 0.5, constant.
    if last:
      self.bias = Tensor.zeros(1, outC, 1, 1) + 0.5
    else:
      self.bias = Tensor.uniform(1, outC, 1, 1)

  def forward(self, x):
    # You might be thinking, "but what about padding?"
    # Answer: Tiling is used to stitch everything back together, though you could pad the image before providing it.
    return x.conv2d(self.weight).add(self.bias)

  def get_parameters(self) -> list:
    return [self.weight, self.bias]

global PEA_CONFIGS
PEA_CONFIGS = {
  "1": {
    "channels": [
      32, 32, 64, 64, 128, 128
    ],
    "noisechk": False
  },
  "2": {
    "channels": [
      48, 48, 48, 48, 48, 48, 48, 48,
      48, 48, 48, 48, 48, 48, 48
    ],
    "noisechk": True
  },
}

class Pea:
  """
  The PEA network.
  PEA is designed for audio upscaling.
  """

  def __init__(self, cfg):
    channels = 2
    self.convs = []
    self.noisechk = cfg["noisechk"]
    for v in cfg["channels"]:
      self.convs.append(Conv3x1Biased(channels, v))
      channels = v
    self.convs.append(Conv3x1Biased(channels, 4, True))

  def context(self):
    """
    Gives the required amount of context.
    This is per-direction - multiply by 2 for total context.
    """
    return len(self.convs)

  def forward(self, x):
    """
    Forward pass: Actually runs the network.
    Input format: (1, 2, 1, X)
    Output format: (1, 4, 1, X - 14)
    (the - 14 represents the 7-frame context border that is lost)
    Note that the first 2 output channels and the second 2 output channels represent separate frames.
    This is so "non-overlapping deconvolution" can be achieved simply through a reshape.
    """
    for v in self.convs:
      x = v.forward(x).leakyrelu(0.1)
    return x

  def get_parameters(self) -> list:
    parameters = []
    for v in self.convs:
      parameters += v.get_parameters()
    return parameters

  def loss(self, out, sample_y):
    # | ||
    # |||_
    standard_loss = (sample_y - out).abs().mean()
    if not self.noisechk:
      return standard_loss
    else:
      # So the main thing I've found is that PEA1 can't handle noise.
      # So this is a little workaround for that.
      out = out.reshape(shape = (2, 2, -1))
      sample_y = sample_y.reshape(shape = (2, 2, -1))
      # (subSample, channel, superSample)
      # remap things around a bit - rather than comparing specific values,
      #  compare the min/max.
      # each of these is a (channel, superSample)
      out_min = (out * -1).max(axis = 0) * -1
      out_max = out.max(axis = 0)
      sample_y_min = (sample_y * -1).max(axis = 0) * -1
      sample_y_max = sample_y.max(axis = 0)
      error_min = (sample_y_min - out_min).abs()
      error_max = (sample_y_max - out_max).abs()
      # Keep some standard loss to try and keep it getting directions right.
      return (error_min + error_max).mean() + (standard_loss * 0.125)

  def forward_tiled(self, image: numpy.ndarray, tile_size: int) -> numpy.ndarray:
    """
    Given audio of the form (N, 2) (NOT a tensor), scales it, pads it, splits it up, forwards the pieces, and reconstitutes it,
     as (N * 2, 2)
    """

    in_samples = image.shape[0]

    # Constant that only really gets repeated a ton here.
    context = self.context()
    context2 = context + context

    image = into_cdat_i(image)

    # Padding next. Note that this padding is done on the whole audio.
    # Padding the tiles would lose critical context, cause seams, etc.
    image = numpy.pad(image, [[0, 0], [0, 0], [0, 0], [context, context]], mode = "edge")

    # Almost resulting output buffer
    image_out = numpy.empty((1, 4, 1, in_samples))

    # Now for tiling.
    # The output tile size is the usable output from an input tile (tile_size).
    # As such, the tiles overlap.
    out_tile_size = tile_size - context2
    for out_x in range(0, image_out.shape[3], out_tile_size):
      print(str(out_x) + " / " + str(image_out.shape[3]))
      # Input is sourced from the same coordinates, but some stuff ought to be
      #  noted here for future reference:
      # + out_x/y's equivalent position w/ the padding is out_x + context.
      # + The output, however, is without context. Input needs context.
      # + Therefore, the input rectangle is expanded on all sides by context.
      # + Therefore, the input position has the context subtracted again.
      # + Therefore:
      in_x = out_x
      # not shown: in_w/in_h = tile_size (as opposed to out_tile_size)
      # Extract tile.
      # Note that numpy will auto-crop this at the bottom-right.
      # This will never be a problem, as tiles are specifically chosen within the padded section.
      tile = image[:, :, :, in_x:in_x + tile_size]
      # Extracted tile dimensions -> output dimensions
      # This is important because of said cropping, otherwise it'd be interior tile size.
      out_w = tile.shape[3] - context2
      # Process tile.
      tile_t = Tensor(tile)
      tile_fwd_t = self.forward(tile_t)
      # Replace tile.
      image_out[:, :, :, out_x:out_x + out_w] = tile_fwd_t.data

    return from_cdat_o(image_out)

# -- Audio Import / Export --
# These functions adjust layout
# 'cdat' here is (1, c, 1, samples)
# where c is either 2 (i) or 4 (o)

def into_cdat_i(image):
  assert len(image.shape) == 2
  assert image.shape[1] == 2
  # Adjust levels (To 0 1)
  image = (image + 1) / 2
  # Rearrange input to (2, N) for expand into reshape
  image = numpy.moveaxis(image, 1, 0)
  # Actually reshape into final (1, 2, 1, N) form
  return image.reshape(1, 2, 1, -1)

def from_cdat_o(image):
  # Adjust levels (To -1 1)
  image = (image * 2) - 1
  # The data (1, 4, 1, in_samples) needs to be rearranged.
  # Into (1, 1, in_samples, 4)
  image = numpy.moveaxis(image, 1, 3)
  # And finish by pulling the upper axis into samples
  return image.reshape(image.shape[2] * 2, 2)

# For training
def into_cdat_o(image):
  assert len(image.shape) == 2
  assert image.shape[1] == 2
  # Adjust levels (To 0 1)
  image = (image + 1) / 2
  # Expand channel count to 4 so it captures pairing
  image = image.reshape(-1, 4)
  # Rearrange input to (4, N) for expand into reshape
  image = numpy.moveaxis(image, 1, 0)
  # Actually reshape into final (1, 4, 1, N) form
  return image.reshape(1, 4, 1, -1)

