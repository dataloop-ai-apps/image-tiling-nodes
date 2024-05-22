import cv2
import numpy as np
import dtlpy as dl
import logging

from multiprocessing.pool import ThreadPool
import concurrent.futures
import os
import tempfile
import math
import sys


logger = logging.getLogger(name='image_tiling')

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from utils import ConstStrideTiles


class TilingBase(dl.BaseServiceRunner):
    """
    Tiling service to split images into tiles.
    """

    def __init__(self):
        self.logger = logger
        self.cycle_status_dict = {}
        self.logger.info('Initializing Image Tiling Service')

    @staticmethod
    def optimal_split(number, target_chunk_size=2000):
        """
        Calculate the optimal chunk size to split a number into parts close to the target chunk size.

        Args:
            number (int): The number to be split.
            target_chunk_size (int): The desired chunk size.

        Returns:
            int: The optimal chunk size.
        """
        if number < target_chunk_size:
            return number

        if number % target_chunk_size == 0:
            return target_chunk_size

        chunks = round(number / target_chunk_size)
        actual_chunk_size = math.ceil(number / chunks)

        if chunks * target_chunk_size < number:
            smaller_chunk_size = math.ceil(number / (chunks + 1))
            if abs(smaller_chunk_size - target_chunk_size) < abs(actual_chunk_size - target_chunk_size):
                actual_chunk_size = smaller_chunk_size

        return actual_chunk_size

    @staticmethod
    def create_tile(image_data, x, y, w, h, out_w, out_h, is_grayscale=False):
        """
        Create a tile from an image.

        Args:
            image_data (np.array): The image data.
            x (int): X coordinate of the tile.
            y (int): Y coordinate of the tile.
            w (int): Width of the tile.
            h (int): Height of the tile.
            out_w (int): Output width of the tile.
            out_h (int): Output height of the tile.
            is_grayscale (bool): Flag indicating if the image is grayscale.

        Returns:
            np.array: The image tile.
        """
        if is_grayscale:
            im_h, im_w = image_data.shape
            tile = np.zeros((h, w), dtype=image_data.dtype)
        else:
            im_h, im_w, im_c = image_data.shape
            tile = np.zeros((h, w, im_c), dtype=image_data.dtype)

        ys = max(y, 0)
        ye = min(y + h, im_h)
        xs = max(x, 0)
        xe = min(x + w, im_w)

        if is_grayscale:
            d = image_data[ys:ye, xs:xe]
        else:
            d = image_data[ys:ye, xs:xe, :]

        xs = 0 if x > 0 else -x
        ys = 0 if y > 0 else -y
        if is_grayscale:
            d_h, d_w = d.shape
            tile[ys:ys + d_h, xs:xs + d_w] = d
        else:
            d_h, d_w, _ = d.shape
            tile[ys:ys + d_h, xs:xs + d_w, :] = d

        out_size = (out_w, out_h)
        if (w, h) != out_size:
            tile = cv2.resize(
                tile, dsize=out_size, interpolation=cv2.INTER_LINEAR if is_grayscale else cv2.INTER_AREA)

        return tile

    def split_image(self, item: dl.Item, context: dl.Context):
        """
        Split the image into tiles.

        Args:
            item (dl.Item): Dataloop item.
            context (dl.Context): Dataloop context, which contains the node with the configuration about tile size and overlapping.

        Returns:
            list: List of parent items with the bounding boxes annotations.
        """
        image_data = cv2.imread(item.download())
        node = context.node
        tile_height = node.metadata['customNodeConfig']['tile_height']
        tile_width = node.metadata['customNodeConfig']['tile_width']
        temp_items_path = tempfile.mkdtemp()
        size = (self.optimal_split(image_data.shape[:2][::-1][0], tile_height),
                self.optimal_split(image_data.shape[:2][::-1][1], tile_width))
        pool = ThreadPool(processes=16)
        async_results = list()
        tiles = ConstStrideTiles(
            image_size=image_data.shape[:2][::-1], tile_size=size, stride=size, include_nodata=False)

        filters = dl.Filters(resource='annotations')
        filters.add(field='type', values='binary')
        annotations = item.annotations.list(filters=filters)

        self.logger.info('Splitting image into {} tiles'.format(len(tiles)))
        for i, (extent, out_size) in enumerate(tiles):
            x, y, w, h = extent
            new_annotations = dl.AnnotationCollection()
            tile = self.create_tile(
                image_data, x, y, w, h, out_size[0], out_size[1])
            for annotation in annotations:
                mask = self.create_tile(
                    annotation.geo, x, y, w, h, out_size[0], out_size[1], is_grayscale=True)
                new_ann = dl.Segmentation(geo=mask, color=annotation.color, label=annotation.label,
                                          attributes=annotation.attributes, description=annotation.description)
                new_annotations.add(new_ann)

            file_path = os.path.join(
                temp_items_path, "{}_{}.jpg".format(item.name.split('.')[0], i))
            cv2.imwrite(file_path, tile)
            async_results.append(
                pool.apply_async(
                    self.upload_item_and_annotations,
                    (file_path, item, new_annotations, x, y)
                )
            )
        pool.close()
        pool.join()

        self.logger.info('Temporary items uploaded to Dataloop')
        items = list()
        for async_result in async_results:
            upload = async_result.get()
            items.append(upload)

        return items

    def upload_item_and_annotations(self, file_path, parent_item: dl.Item, new_annotations, x, y):
        """
        Upload a tile and its annotations to the dataset.

        Args:
            file_path (str): Path to the tile image.
            parent_item (dl.Item): Parent item.
            new_annotations (dl.AnnotationCollection): Annotations for the tile.
            x (int): X coordinate of the tile in the original image.
            y (int): Y coordinate of the tile in the original image.

        Returns:
            dl.Item: Uploaded item.
        """
        new_item = parent_item.dataset.items.upload(local_path=file_path, remote_path="/tiles", item_metadata={
            "user": {
                "parentItemId": parent_item.id,
                "originalTop": y,
                "originalLeft": x,
            }}, overwrite=True)
        new_item.annotations.upload(annotations=new_annotations)
        return new_item

    @staticmethod
    def process_item(item, parent_item):
        """
        Process an item to update the annotations in the parent item.

        Args:
            item (dl.Item): Tile item.
            parent_item (dl.Item): Parent item.

        Returns:
            dict: Dictionary of processed annotations.
        """
        original_left = item.metadata['user']['originalLeft']
        original_top = item.metadata['user']['originalTop']
        filters = dl.Filters(resource='annotations')
        filters.add(field='type', values='binary')
        item_annotations = item.annotations.list(filters=filters)
        item_results = {}
        for annotation in item_annotations:
            if annotation.label not in item_results:
                item_results[annotation.label] = []
            new_geo = np.zeros((parent_item.height, parent_item.width))
            new_geo[original_top:original_top + annotation.geo.shape[0],
                    original_left:original_left + annotation.geo.shape[1]] = annotation.geo
            item_results[annotation.label].append(new_geo)
        return item_results

    def add_mask_to_parent(self, parent_item: dl.Item):
        """
        Combine the mask annotations from all tiles and add/update them in the parent item.

        Args:
            parent_item (dl.Item): Parent item.
        """
        custom_filter = {
            'page': 0,
            'pageSize': 0,
            'resource': 'items',
            'filter': {'$and': [{'hidden': False},
                                {'type': 'file'},
                                {'metadata.user.parentItemId': parent_item.id}]},
        }
        filters = dl.Filters(custom_filter=custom_filter)
        dataset = parent_item.dataset
        items = dataset.items.list(filters=filters).all()
        all_annotations = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                self.process_item, item, parent_item) for item in items]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for label, geos in result.items():
                    if label not in all_annotations:
                        all_annotations[label] = []
                    all_annotations[label].extend(geos)

        new_annotations = dl.AnnotationCollection()
        for key, value in all_annotations.items():
            filters = dl.Filters(resource='annotations')
            filters.add(field='type', values='binary')
            filters.add(field='label', values=key)
            parent_annotations = parent_item.annotations.list(filters=filters)
            if len(parent_annotations) == 0:
                print(f'Creating annotation for {key}')
                new_geo = np.any(value, axis=0)
                new_annotations.add(dl.Segmentation(geo=new_geo, label=key))
            else:
                print(f'Updating annotation for {key}')
                parent_annotation = parent_annotations[0]
                parent_annotation.geo = np.any(
                    [*value, parent_annotation.geo], axis=0)
                parent_annotation.update()

        parent_item.annotations.upload(annotations=new_annotations)

        return parent_item
