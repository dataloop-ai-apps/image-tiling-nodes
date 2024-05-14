import cv2
import numpy as np
import dtlpy as dl
import logging
from tiling import ConstSizeTiles
from multiprocessing.pool import ThreadPool
import os
import tempfile
import time
from rtree import index
from shapely.geometry import Polygon


logger = logging.getLogger(name='image_tiling')


class TilingBase(dl.BaseServiceRunner):
    """
    Tiling service to split images into tiles.
    """

    def __init__(self):
        self.logger = logger
        self.logger.info('Initializing Image Tiling Service')

    @staticmethod
    def read_data(image_data, x, y, w, h, out_w, out_h):
        """
        Create a tile from an image

        Args:
            image_data: image data
            x: x coordinate of the tile
            y: y coordinate of the tile
            w: width of the tile
            h: height of the tile
            out_w: output width of the tile
            out_h: output height of the tile

        Returns:
            Tile
        """

        im_h, im_w, im_c = image_data.shape
        tile = np.zeros((h, w, im_c), dtype=image_data.dtype)

        ys = max(y, 0)
        ye = min(y + h, im_h)
        xs = max(x, 0)
        xe = min(x + w, im_w)
        d = image_data[ys:ye, xs:xe, :]

        xs = 0 if x > 0 else -x
        ys = 0 if y > 0 else -y
        d_h, d_w, _ = d.shape
        tile[ys:ys + d_h, xs:xs + d_w, :] = d

        out_size = (out_w, out_h)
        if (w, h) != out_size:
            tile = cv2.resize(tile, dsize=out_size)

        return tile

    def split_image(self, item: dl.Item, context: dl.Context):
        """
        Split the image into tiles.

        Args:
            item: dataloop item
            context: dataloop context, which contains the node with the configuration about tile size and overlapping

        Returns:
            parent item with the bounding boxes annotations
        """

        image_data = cv2.imread(item.download())
        node = context.node
        tile_size = node.metadata['customNodeConfig']['tile_size']
        min_overlapping = node.metadata['customNodeConfig']['min_overlapping']
        temp_items_path = tempfile.mkdtemp()
        pool = ThreadPool(processes=16)
        async_results = list()
        try:
            tiles = ConstSizeTiles(image_size=image_data.shape[:2][::-1], tile_size=tile_size, min_overlapping=min_overlapping)
        except ValueError as e:
            self.logger.warning('Image is smaller than tile size, skipping')
            if 'user' not in item.metadata:
                item.metadata['user'] = dict()
            item.metadata['user']['parentItemId'] = item.id
            item.metadata['user']['originalTop'] = 0
            item.metadata['user']['originalLeft'] = 0
            item.update()
            return [item]

        self.logger.info('Splitting image into {} tiles'.format(len(tiles)))
        for i, (extent, out_size) in enumerate(tiles):
            x, y, w, h = extent
            tile = self.read_data(image_data, x, y, w, h, out_size[0], out_size[1])

            file_path = os.path.join(temp_items_path, "{}_{}.jpg".format(item.name.split('.')[0], i))
            cv2.imwrite(file_path, tile)
            async_results.append(
                pool.apply_async(
                    item.dataset.items.upload,
                    kwds={
                        "local_path": file_path,
                        "remote_path": '.dataloop_temp',
                        "overwrite": True,
                        "item_metadata": {
                            "user": {
                                "parentItemId": item.id,
                                "originalTop": y,
                                "originalLeft": x,
                            }
                        },
                    },
                )
            )
        pool.close()
        pool.join()

        self.logger.info('Temporary items uploaded to Dataloop')
        items = list()
        for async_result in async_results:
            upload = async_result.get()
            items.append(upload)

        item.metadata['tiling'] = dict()
        item.metadata['tiling']['total'] = len(items)
        item.metadata['tiling']['processed'] = 0
        item.update()

        return items

    @staticmethod
    def wait_for_cycle(item: dl.Item, context: dl.Context, progress: dl.Progress):
        parent_item_id = item.metadata['user']['parentItemId']
        parent_item = item.dataset.items.get(item_id=parent_item_id)
        parent_item.metadata['tiling']['processed'] += 1
        if parent_item.metadata['tiling']['processed'] == parent_item.metadata['tiling']['total']:
            progress.update(action="continue")
        else:
            progress.update(action="wait")
        parent_item.update()
        return parent_item

    @staticmethod
    def same_object(box1, box2, trashold=0.7):
        """
        Check if two bounding boxes are the same object

        Args:
            box1: first bounding box
            box2: second bounding box
            trashold: the IOU trashold to consider the boxes the same object

        Returns:
            True if the boxes are the same object, False otherwise
        """
        box_1 = [[box1['x1'], box1['y1']], [box1['x2'], box1['y1']], [box1['x2'], box1['y2']], [box1['x1'], box1['y2']]]
        box_2 = [[box2['x1'], box2['y1']], [box2['x2'], box2['y1']], [box2['x2'], box2['y2']], [box2['x1'], box2['y2']]]
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)

        iou1 = poly_1.intersection(poly_2).area / poly_1.area
        iou2 = poly_1.intersection(poly_2).area / poly_2.area
        return max(iou1, iou2) > trashold

    @staticmethod
    def bounding_box_area(box):
        """
        Calculate the area of a bounding box

        Args:
            box: bounding box

        Returns:
            Area of the bounding box
        """

        return (box['x2'] - box['x1']) * (box['y2'] - box['y1'])

    @staticmethod
    def process_item(item_id):
        """
        Process and delete an item and return all bounding boxes annotations

        Args:
            item_id: item id

        Returns:
            All bounding boxes annotations : dict
        """
        item = dl.items.get(item_id=item_id)
        annotations = item.annotations.list()
        x = item.metadata['user']['originalLeft']
        y = item.metadata['user']['originalTop']

        local_bounding_boxes = {}
        for annotation in annotations:
            key = annotation.label
            box = {"label": key, 'x1': annotation.left + x, 'y1': annotation.top + y, 'x2': annotation.right + x, 'y2': annotation.bottom + y, 'metadata': annotation.metadata}
            if key in local_bounding_boxes:
                local_bounding_boxes[key].append(box)
            else:
                local_bounding_boxes[key] = [box]

        item.delete()
        return local_bounding_boxes

    def add_to_parent_item(self, item: dl.Item):
        parent_item_id = item.id
        parent_item = item
        filters = dl.Filters()
        filters.add(field='metadata.user.parentItemId', values=parent_item_id)
        dataset = item.dataset
        items = dataset.items.list(filters=filters).all()
        ids = [item.id for item in items]

        pool = ThreadPool(processes=6)
        results = [pool.apply_async(self.process_item, (item_id,)) for item_id in ids]
        bounding_boxes = {}

        for result in results:
            local_boxes = result.get()
            for key, boxes in local_boxes.items():
                if key in bounding_boxes:
                    bounding_boxes[key].extend(boxes)
                else:
                    bounding_boxes[key] = boxes

        pool.close()
        pool.join()

        self.logger.info('Prediction processed and items deleted')

        idx_dict = {}
        for label, boxes in bounding_boxes.items():
            idx_dict[label] = index.Index()
            for i, box in enumerate(boxes):
                # The bounding box format is (min_x, min_y, max_x, max_y)
                idx_dict[label].insert(i, (box['x1'], box['y1'], box['x2'], box['y2']))

        self.logger.info('Bounding boxes indexed')

        for label, boxes in bounding_boxes.items():
            final_boxes = boxes.copy()
            for i, box in enumerate(boxes):
                # Query for nearby boxes
                nearby_boxes = list(idx_dict[label].intersection((box['x1'], box['y1'], box['x2'], box['y2'])))
                for j in nearby_boxes:
                    if i != j and self.same_object(boxes[i], boxes[j]):
                        if self.bounding_box_area(boxes[i]) > self.bounding_box_area(boxes[j]):
                            try:
                                final_boxes.remove(boxes[j])
                            except ValueError:
                                pass

            bounding_boxes[label] = final_boxes

        self.logger.info('Bounded boxes merged and deduplicated')

        new_annotations = dl.AnnotationCollection()
        for label, boxes in bounding_boxes.items():
            for box in boxes:
                new_ann = dl.Box(top=box['y1'], left=box['x1'], bottom=box['y2'], right=box['x2'], label=label)
                new_annotations.add(new_ann, metadata=box.get('metadata', {}))

        parent_item.annotations.upload(annotations=new_annotations)
        self.logger.info('Bounding boxes annotations uploaded to Parent Item')
        return parent_item
