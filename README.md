# Smart tiling pipeline nodes

## Installation

To install the Dataloop Image Tiling Service from the Dataloop Marketplace, follow these steps:

1. **Sign in to Dataloop:**
   Go to the Dataloop platform and sign in with your credentials.

2. **Navigate to the Marketplace:**
   In the Dataloop dashboard, navigate to the Marketplace section.

3. **Find the Image Tiling Service:**
   Search for the "Image Tiling Service" in the Marketplace.

4. **Install the Service:**
   Click on the service and follow the on-screen instructions to install it to your Dataloop environment.

## Pipeline Nodes

The installation will create the following pipeline nodes:

1. **Split Image:**

   - **Description:** Split image to tiles based on tile size and minimal overlap
   - **Configuration:**
     - `tile_size`: The size of each tile in pixels.
     - `min_overlapping`: The minimum overlap between adjacent tiles in pixels.

2. ## **Wait For Cycle:**

   - **Description:** Wait until all previous executions are done

3. **Add Annotations To Main Item:**

   - **Description:** Add all annotations to the main item
