# -*- coding:utf-8 -*-
from six import with_metaclass
from abc import ABCMeta, abstractmethod
import logging
import math

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


logger = logging.getLogger('tiling')


class BaseTiles(with_metaclass(ABCMeta, object)):
    """
    Base class to tile an image.
    See the implementations
        - ConstSizeTiles
        - ConstStrideTiles
    """

    def __init__(self, image_size, tile_size=(128, 128), scale=1.0):
        """Initialize tiles

        Args:
            image_size (list/tuple): input image size in pixels
            tile_size (int or list/tuple): output tile size in pixels
            scale (float): tile scaling factor
        """

        if not (isinstance(image_size, Sequence) and len(image_size) == 2):
            raise TypeError("Argument image_size should be (sx, sy)")
        for s in image_size:
            if s < 1:
                raise ValueError("Values of image_size should be positive")

        if not (isinstance(tile_size, int) or (isinstance(tile_size, Sequence) and len(tile_size) == 2)):
            raise TypeError("Argument tile_size should be either int or pair of integers (sx, sy)")
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        for s in tile_size:
            if s < 1:
                raise ValueError("Values of tile_size should be positive")

        if scale <= 0:
            raise ValueError("Argument scale should be positive")

        for tile_dim, img_dim in zip(tile_size, image_size):
            if int(tile_dim / scale) > img_dim:
                raise ValueError("Scale {} and tile size {} should not be larger "
                                 "than image size {}".format(scale, tile_dim, img_dim))

        self.image_size = image_size
        self.tile_size = tile_size
        self.scale = float(scale)
        # Apply floor to tile extent (tile size / scale)
        # Output size is then ceil(extent * scale), extent is <= tile_extent
        # ceil(extent * scale) < ceil(tile_extent * scale) = ceil(floor(tile_extent / scale) * scale)<= tile_size
        self.tile_extent = [int(math.floor(d / self.scale)) for d in self.tile_size]
        self._index = 0
        self._max_index = 0

    @abstractmethod
    def __len__(self):
        """Method to get total number of tiles
        """

    @abstractmethod
    def __getitem__(self, idx):
        """Method to get the tile at index

        Args:
            idx: (int)
        """

    def next(self):
        """Method to get next tile

        Returns:
            tile data (ndarray), tile extent (list) in the original image, in pixels
        """
        if self._index < 0 or self._index >= self._max_index:
            raise StopIteration()

        res = self[self._index]
        # ++
        self._index += 1

        return res

    __next__ = next


def ceil_int(x):
    return int(math.ceil(x))


class ConstStrideTiles(BaseTiles):
    """Class provides tile parameters (offset, extent) to extract data from image.

    Examples:

        .. code-block:: python

            from tiling import ConstStrideTiles

            tiles = ConstStrideTiles(image_size=(500, 500), tile_size=(256, 256), stride=(100, 100),
                                     origin=(-100, -100),
                                     scale=1.0,
                                     include_nodata=True)

            print("Number of tiles: %i" % len(tiles))
            for extent, out_size in tiles:
                x, y, width, height = extent
                data = read_data(x, y, width, height,
                                 out_width=out_size[0],
                                 out_height=out_size[1])
                print("data.shape: {}".format(data.shape))

    Args:
        image_size (list/tuple of int): input image size in pixels (width, height)
        tile_size (int or list/tuple of int): output tile size in pixels (width, height)
        stride (list/tuple of int): horizontal and vertical strides in pixels.
            Values need to be positive larger than 1 pixel. Stride value is impacted with scale and corresponds
            to a sliding over scaled image.
        scale (float): Scaling applied to the input image parameters before extracting tile's extent
        origin (list or tuple of int): point in pixels in the original image from where to start the tiling.
            Values can be positive or negative.
        include_nodata (bool): Include or not nodata. If nodata is included then tile extents have all the
            same size, otherwise tiles at boundaries will be reduced
    """

    def __init__(self, image_size, tile_size, stride=(1, 1), scale=1.0, origin=(0, 0), include_nodata=True):
        super(ConstStrideTiles, self).__init__(image_size=image_size, tile_size=tile_size, scale=scale)

        if not (isinstance(stride, int) or (isinstance(stride, Sequence) and len(stride) == 2)):
            raise TypeError("Argument stride should be (sx, sy)")

        if isinstance(stride, int):
            stride = (stride, stride)

        # Apply scale on the stride
        stride = [int(math.floor(s / self.scale)) for s in stride]
        for v in stride:
            if v < 1:
                raise ValueError("Scaled stride values `floor(stride / scale)` should be larger than 1 pixel")

        self.stride = stride
        self.origin = origin
        self.include_nodata = include_nodata
        self.nx = ConstStrideTiles._compute_number_of_tiles(self.image_size[0], self.tile_extent[0],
                                                            self.origin[0], self.stride[0])
        self.ny = ConstStrideTiles._compute_number_of_tiles(self.image_size[1], self.tile_extent[1],
                                                            self.origin[1], self.stride[1])
        self._max_index = self.nx * self.ny

    def __len__(self):
        """Method to get total number of tiles
        """
        return self._max_index

    @staticmethod
    def _compute_tile_extent(idx, tile_extent, stride, origin, image_size, include_nodata):
        """Method to compute tile extent: offset, extent for a given index
        """

        offset = idx * stride + origin
        if not include_nodata:
            extent = max(offset + tile_extent, 0) - max(offset, 0)
            extent = min(extent, image_size - offset)
            offset = max(offset, 0)
        else:
            extent = tile_extent
        return offset, extent

    @staticmethod
    def _compute_out_size(computed_extent, tile_extent, tile_size, scale):
        """Method to compute tile output size for a given computed extent.
        """
        if computed_extent < tile_extent:
            return ceil_int(1.0 * computed_extent * scale)
        return tile_size

    def __getitem__(self, idx):
        """Method to get the tile at index `idx`

        Args:
            idx: (int) tile index between `0` and `len(tiles)`

        Returns:
            (tuple) tile extent, output size in pixels

        Tile extent in pixels: x offset, y offset, x tile extent, y tile extent.
        If scale is 1.0, then x tile extent, y tile extent are equal to tile size
        Output size in pixels: output width, height. If include_nodata is False and other parameters are such that
        tiles can go outside the image, then tile extent and output size are cropped at boundaries.
        Otherwise, output size is equal the input tile size.
        """
        if idx < -self._max_index or idx >= self._max_index:
            raise IndexError("Index %i is out of ranges %i and %i" % (idx, 0, self._max_index))

        idx = idx % self._max_index  # Handle negative indexing as -1 is the last
        x_index = idx % self.nx
        y_index = int(idx * 1.0 / self.nx)

        x_offset, x_extent = self._compute_tile_extent(x_index, self.tile_extent[0],
                                                       self.stride[0], self.origin[0], self.image_size[0],
                                                       self.include_nodata)
        y_offset, y_extent = self._compute_tile_extent(y_index, self.tile_extent[1],
                                                       self.stride[1], self.origin[1], self.image_size[1],
                                                       self.include_nodata)
        x_out_size = self.tile_size[0] if self.include_nodata else \
            self._compute_out_size(x_extent, self.tile_extent[0], self.tile_size[0], self.scale)
        y_out_size = self.tile_size[1] if self.include_nodata else \
            self._compute_out_size(y_extent, self.tile_extent[1], self.tile_size[1], self.scale)
        return (x_offset, y_offset, x_extent, y_extent), (x_out_size, y_out_size)

    @staticmethod
    def _compute_number_of_tiles(image_size, tile_extent, origin, stride):
        """Method to compute number of overlapping tiles
        """
        max_extent = max(tile_extent, stride)
        return max(ceil_int(1 + (image_size - max_extent - origin) * 1.0 / stride), 1)


class ConstSizeTiles(BaseTiles):
    """Class provides constant size tile parameters (offset, extent) to extract data from image.
    Generated tile extents can overlap and do not includes nodata paddings.

    Examples:

        .. code-block:: python

            from tiling import ConstSizeTiles

            tiles = ConstSizeTiles(image_size=(500, 500), tile_size=(256, 256), min_overlapping=15, scale=1.0)

            print("Number of tiles: %i" % len(tiles))
            for extent, out_size in tiles:
                x, y, width, height = extent
                data = read_data(x, y, width, height,
                                 out_width=out_size[0],
                                 out_height=out_size[1])
                print("data.shape: {}".format(data.shape))

    Args:
        image_size (list/tuple of int): input image size in pixels (width, height)
        tile_size (int or list/tuple of int): output tile size in pixels (width, height)
        min_overlapping (int): minimal overlapping in pixels between tiles.
        scale (float): Scaling applied to the input image parameters before extracting tile's extent
    """

    def __init__(self, image_size, tile_size, min_overlapping=0, scale=1.0):
        super(ConstSizeTiles, self).__init__(image_size=image_size, tile_size=tile_size, scale=scale)

        if not (0 <= min_overlapping < min(self.tile_extent[0], self.tile_extent[1])):
            raise ValueError("Argument min_overlapping should be between 0 and min tile_extent = tile_size / scale"
                             ", but given {}".format(min_overlapping))

        self.min_overlapping = min_overlapping
        # Compute number of tiles:
        self.nx = ConstSizeTiles._compute_number_of_tiles(self.tile_extent[0], self.image_size[0], min_overlapping)
        self.ny = ConstSizeTiles._compute_number_of_tiles(self.tile_extent[1], self.image_size[1], min_overlapping)

        if self.nx < 1:
            raise ValueError("Number of horizontal tiles is not positive. Wrong input parameters: {}, {}, {}".format(
                self.tile_extent[0], self.image_size[0], min_overlapping))

        if self.ny < 1:
            raise ValueError("Number of vertical tiles is not positive. Wrong input parameters: {}, {}, {}".format(
                self.tile_extent[1], self.image_size[1], min_overlapping))

        self.float_overlapping_x = ConstSizeTiles._compute_float_overlapping(self.tile_extent[0],
                                                                             self.image_size[0], self.nx)
        self.float_overlapping_y = ConstSizeTiles._compute_float_overlapping(self.tile_extent[1],
                                                                             self.image_size[1], self.ny)
        self._max_index = self.nx * self.ny

    def __len__(self):
        """Method to get total number of tiles
        """
        return self._max_index

    @staticmethod
    def _compute_tile_extent(idx, tile_extent, overlapping):
        """Method to compute tile extent: offset, extent for a given index
        """

        offset = int(round(idx * (tile_extent - overlapping)))
        return offset, int(round(tile_extent))

    def __getitem__(self, idx):
        """Method to get the tile at index `idx`

        Args:
            idx: (int) tile index between `0` and `len(tiles)`

        Returns:
            (tuple) tile extent, output size in pixels

        If scale is 1.0, then x tile extent, y tile extent are equal to tile size
        """
        if idx < -self._max_index or idx >= self._max_index:
            raise IndexError("Index %i is out of ranges %i and %i" % (idx, 0, self._max_index))

        idx = idx % self._max_index  # Handle negative indexing as -1 is the last
        x_tile_index = idx % self.nx
        y_tile_index = int(idx * 1.0 / self.nx)

        x_tile_offset, x_tile_extent = self._compute_tile_extent(x_tile_index, self.tile_extent[0],
                                                                 self.float_overlapping_x)
        y_tile_offset, y_tile_extent = self._compute_tile_extent(y_tile_index, self.tile_extent[1],
                                                                 self.float_overlapping_y)
        return (x_tile_offset, y_tile_offset, x_tile_extent, y_tile_extent), (self.tile_size[0], self.tile_size[1])

    @staticmethod
    def _compute_number_of_tiles(tile_extent, image_size, min_overlapping):
        """Method to compute number of overlapping tiles for a given image size
        """
        return ceil_int(image_size * 1.0 / (tile_extent - min_overlapping + 1e-10))

    @staticmethod
    def _compute_float_overlapping(tile_size, image_size, n):
        """Method to float overlapping
        """
        return (tile_size * n - image_size) * 1.0 / (n - 1.0) if n > 1 else 0
