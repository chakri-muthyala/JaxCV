from abc import abstractmethod
from typing import Union, List, Tuple

import math
import jax
import jax.numpy as jnp
from einops import rearrange
import warnings

from .base import Transformation, InputType, same_type


class ImageLevelTransformation(Transformation):
    pass


class GridShuffle(ImageLevelTransformation):
    """Divides the image into grid cells and shuffles them randomly.

    Args:
        grid_size (int, int): Tuple of `(gridcells_x, gridcells_y)` that specifies into how many
            cells the image is to be divided along each axis.
            If only a single number is given, that value will be used along both axes.
            Currently requires that each image dimension is a multiple of the corresponding value.
        p (float): Probability of applying the transformation
    """
    def __init__(self, grid_size: Union[Tuple[int, int], int] = (4, 4), p: float = 0.5, input_types=[InputType.IMAGE]):
        super().__init__(input_types)
        if hasattr(grid_size, '__iter__'):
            self.grid_size = tuple(grid_size)
        else:
            self.grid_size = (self.grid_size, self.grid_size)
        self.grid_size = grid_size
        self.probability = p


    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType]=None, invert=False) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        key1, key2 = jax.random.split(rng)
        do_apply = jax.random.bernoulli(key1, self.probability)
        val = []
        for input, type in zip(inputs, input_types):
            current = None
            if same_type(type, InputType.IMAGE) or same_type(type, InputType.MASK) or same_type(type, InputType.DENSE):
                raw_image = input

                H, W, C = raw_image.shape
                gx, gy = self.grid_size

                if H % self.grid_size[0] != 0:
                    raise ValueError(f"Image height ({H}) needs to be a multiple of gridcells_y ({gy})")
                if W % self.grid_size[1] != 0:
                    raise ValueError(f"Image width ({W}) needs to be a multiple of gridcells_x ({gx})")

                image = rearrange(raw_image, '(gy h) (gx w) c -> (gy gx) h w c', gx=gx, gy=gy)
                if invert:
                    inv_permutation = jnp.argsort(jax.random.permutation(key2, image.shape[0]))
                    image = image[inv_permutation]
                else:
                    image = jax.random.permutation(key2, image)
                image = rearrange(image, '(gy gx) h w c -> (gy h) (gx w) c', gx=gx, gy=gy)
                current = jnp.where(do_apply, image, raw_image)
            else:
                raise NotImplementedError(f"GridShuffle for {type} not yet implemented")
                current = input
            val.append(current)
        return val


class _ConvolutionalBlur(ImageLevelTransformation):
    @abstractmethod
    def __init__(self, p: float = 0.5, input_types=[InputType.IMAGE]):
        super().__init__(input_types)
        self.probability = p
        self.kernel = None
        self.kernelsize = -1

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType]=None, invert=False) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        val = []
        do_apply = jax.random.bernoulli(rng, self.probability)
        p0 = self.kernelsize // 2
        p1 = self.kernelsize - p0 - 1
        for input, type in zip(inputs, input_types):
            current = None
            if same_type(type, InputType.IMAGE):
                if invert:
                    warnings.warn("Trying to invert a Blur Filter, which is not invertible.")
                    current = input
                else:
                    image_padded = jnp.pad(input, [(p0, p1), (p0, p1), (0, 0)], mode='edge')
                    image_padded = rearrange(image_padded, 'h w (c c2) -> c c2 h w', c2=1)
                    convolved = jax.lax.conv(image_padded, self.kernel, [1, 1], 'valid')
                    convolved = rearrange(convolved, 'c c2 h w -> h w (c c2)', c2=1)
                    current = jnp.where(do_apply, convolved, input)
            else:
                current = input
            val.append(current)
        return val


class Blur(_ConvolutionalBlur):
    def __init__(self, size: int = 5, p: float = 0.5):
        super().__init__(p)
        self.kernel = jnp.ones([1, 1, size, size])
        self.kernel = self.kernel / self.kernel.sum()
        self.kernelsize = size


class GaussianBlur(_ConvolutionalBlur):
    def __init__(self, sigma: int = 3, p: float = 0.5):
        super().__init__(p)
        N = int(math.ceil(2 * sigma))
        rng = jnp.linspace(-2.0, 2.0, N)
        x = rng.reshape(1, -1)
        y = rng.reshape(-1, 1)

        self.kernel = jnp.exp((-0.5/sigma) * (x*x + y*y))
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel.reshape(1, 1, N, N)
        self.kernelsize = N
