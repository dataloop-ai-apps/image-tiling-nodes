import unittest
import types
import dtlpy as dl
import sys
import os
import cv2
import numpy as np
import unittest.mock

from nodes.smart_tiling.split_image import TilingBase


class TestSplitImage(unittest.TestCase):

    def test_split_image_logic(self):
        # --- Mock Dataloop and environment setup ---
        # Comment out or replace with actual setup if integration testing is needed
        # dl.setenv('<env>')
        # if dl.token_expired():
        #     dl.login()
        # dataset: dl.Dataset = dl.datasets.get(dataset_id='<dataset_id>')
        # item: dl.Item = dataset.items.get(item_id='<item_id>')

        # Basic Mocking for Dataloop Item
        mock_item = unittest.mock.MagicMock(spec=dl.Item)
        mock_item.id = 'mock_item_id_123'
        mock_item.name = 'test_image.jpg'
        mock_item.download.return_value = './dummy_test_image.jpg' # Path for a dummy image
        mock_item.to_json.return_value = {
            'metadata': {
                'system': {},
                'user': {
                    "hello": "world"
                },
                "random": "random"
            }
        }
        # Mock the dataset attribute and its items.upload method
        mock_item.dataset = unittest.mock.MagicMock()

        # Configure the mock for the uploaded item to have metadata
        mock_uploaded_item = unittest.mock.MagicMock(spec=dl.Item)
        mock_uploaded_item.metadata = {
            'user': {
                'originalLeft': 0,  # Provide some default mock values
                'originalTop': 0
            }
        }
        mock_item.dataset.items.upload.return_value = mock_uploaded_item # Mock uploaded item

        # --- Mock Node Configuration ---
        mock_node_metadata = {
            'customNodeConfig': {
                'tile_size': 500,
                'crop_type': 'crop_with_annotations', # or 'crop_without_annotations'
                'copy_original_metadata': 'true',
                'min_overlapping': 50
            }
        }
        mock_node = types.SimpleNamespace(metadata=mock_node_metadata)
        mock_context = types.SimpleNamespace(node=mock_node)

        # --- TilingBase Instantiation ---
        tiling_base = TilingBase()

        # --- Create a dummy image file for cv2.imread ---
        dummy_image_data = np.zeros((600, 600, 3), dtype=np.uint8) # Ensure image is larger than tile_size
        cv2.imwrite(mock_item.download.return_value, dummy_image_data)

        # --- Execute the method under test ---
        try:
            items = tiling_base.split_image(mock_item, mock_context)

            # --- Assertions (customize these based on expected behavior) ---
            self.assertIsNotNone(items, "split_image should return a list of items.")
            self.assertGreater(len(items), 0, "Should create at least one tile.")

            # Verify that item.dataset.items.upload was called
            mock_item.dataset.items.upload.assert_called()

            # Example: Check metadata of the first uploaded tile
            if items and mock_item.dataset.items.upload.called:
                # Get the arguments from the first call to upload
                _, kwargs = mock_item.dataset.items.upload.call_args_list[0]
                self.assertIn('item_metadata', kwargs)
                uploaded_metadata = kwargs['item_metadata']
                self.assertEqual(uploaded_metadata['user']['parentItemId'], mock_item.id)
                self.assertIn('originalTop', uploaded_metadata['user'])
                self.assertIn('originalLeft', uploaded_metadata['user'])

            # TODO: Add more specific assertions:
            # - Check the number of tiles created based on tile_size, image_size, and min_overlapping.
            # - If crop_type is 'crop_with_annotations', verify annotations on tiles.
            # - Verify the content/names of the files passed to item.dataset.items.upload.

        finally:
            # --- Clean up dummy image file ---
            if os.path.exists(mock_item.download.return_value):
                os.remove(mock_item.download.return_value)


if __name__ == '__main__':
    unittest.main() 