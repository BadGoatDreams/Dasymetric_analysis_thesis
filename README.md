Markdown

# Python Dasymetric Mapping Utilities

This repository contains Python code for performing weighted dasymetric mapping, primarily aimed at disaggregating population data (or other count data) from source zones (e.g., census tracts) to target zones (e.g., building footprints).

## Description

Dasymetric mapping refines the spatial distribution of data by using ancillary information (like building locations, land use types, etc.) to allocate counts more accurately than simple area weighting. This project provides:

1.  A flexible Python function (`weighted_dasymetric_flexible` in `dasy_utils.py`) attempting to implement various dasymetric techniques.
2.  Example scripts demonstrating usage, particularly highlighting the use of the `tobler` library as a robust alternative due to issues encountered with the custom function's core dependency.

The primary goal during development was to allocate Benton County teenage population data from census tracts to building footprints.

## Features (of the custom `weighted_dasymetric_flexible` function)

* **Multiple Weighting Sources:** Can incorporate weights based on target attributes (e.g., number of floors), ancillary vector layers (e.g., land use factors), and ancillary raster layers (e.g., population density).
* **Area Weighting:** Can include the intersected area as part of the weight calculation.
* **Binary Masking:** Allows excluding certain target areas based on overlap with a mask layer (e.g., parks, water bodies).
* **Integer Output:** Includes an option (`integerize=True`) to use the Largest Remainder Method (LRM) to produce sum-preserving integer outputs.
* **Automatic Saving:** Includes an option (`save_output=True`) to save results directly to an ESRI Shapefile.

## Current Status & Limitations

* **`geopandas.overlay` Issue:** Extensive testing revealed a persistent issue where the core `geopandas.overlay(..., how='intersection')` operation, used within the `weighted_dasymetric_flexible` function, consistently returns an empty result for the provided Benton County test data, despite visual confirmation of overlap and successful geometry validation/fixing attempts (`buffer(0)`). This prevents the custom function from calculating intersections and weights correctly in its current state for this specific dataset/environment.
* **`tobler` Library Recommended:** Due to the `overlay` issue, the recommended approach for performing the dasymetric allocation for this project is to use the specialized `tobler` library. Example scripts using `tobler` (e.g., `barebones_example_tobler_manual_lrm.py`) have been shown to successfully produce non-zero float results.
* **Integerization:** While the custom function includes built-in LRM, the current working approach uses `tobler`'s float output, potentially followed by manual LRM steps (as shown in `barebones_example_tobler_manual_lrm.py`) or simple rounding if sum preservation is not strictly required.

## Requirements

* Python 3.x
* GeoPandas
* Pandas
* NumPy
* Shapely
* **Tobler:** Required for the currently recommended working examples.
* **Optional (for raster features/plotting):**
    * Rasterio
    * Rasterstats
    * Matplotlib

You can install the core requirements using pip:

```bash
pip install geopandas pandas numpy tobler shapely
# Optional for raster/plotting:
# pip install rasterio rasterstats matplotlib

Usage
Recommended Approach (Using tobler + Manual LRM)

This approach uses tobler for the reliable area-weighted interpolation and then applies the Largest Remainder Method manually to achieve sum-preserving integer results.

See the script barebones_example_tobler_manual_lrm.py for a full example. Key steps:
Python

import geopandas as gpd
import pandas as pd
import os
import math
import numpy as np
from tobler.area_weighted import area_interpolate
# Assume _integerize_lrm helper function is defined in the script (copied from dasy_utils.py)

# 1. Define Paths and Parameters
base_data_dir = "C:/Users/plato/Documents/Dasymetric_analysis_thesis/dasy_test_gemini/"
source_filepath = os.path.join(base_data_dir, "Teen_pop_benton/benton_census_tracts_teen_pop_short.shp")
target_filepath = os.path.join(base_data_dir, "residences_perhaps/residences_perhaps.shp")
output_dir = os.path.join(base_data_dir, "output_shapefiles")
output_filename_shp = "teen_tobler_lrm_integer.shp"
source_pop_col = "tot_teen"
source_id_col = "GEOID" # Ensure this column exists in source data
output_col_name_int = "teen_lrm" # <= 10 chars

# 2. Load and Prepare Data (Load, Set CRS, Reproject to Projected CRS, buffer(0))
print("Loading and preparing data...")
source_data = gpd.read_file(source_filepath).set_crs("EPSG:4269", allow_override=True).to_crs("EPSG:2991")
target_data = gpd.read_file(target_filepath).set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:2991")
source_data['geometry'] = source_data.geometry.buffer(0)
target_data['geometry'] = target_data.geometry.buffer(0)
source_data = source_data[~source_data.geometry.is_empty]
target_data = target_data[~target_data.geometry.is_empty]
source_data[source_pop_col] = pd.to_numeric(source_data[source_pop_col], errors='coerce').fillna(0)
original_total_overall = source_data[source_pop_col].sum()

# 3. Tobler Area Interpolation (Float results)
print("Running Tobler area interpolation...")
interpolated_float_gdf = area_interpolate(
    source_df=source_data[[source_id_col, source_pop_col, 'geometry']],
    target_df=target_data,
    extensive_variables=[source_pop_col]
)

# 4. Re-link Source Info for LRM
print("Re-linking source info...")
target_for_join = interpolated_float_gdf[[source_pop_col, 'geometry']].copy()
target_for_join['__target_id'] = target_for_join.index
source_orig_pop_col = f"__{source_pop_col}_orig"
source_for_join = source_data[[source_id_col, source_pop_col, 'geometry']].rename(columns={source_pop_col: source_orig_pop_col})
joined_gdf = gpd.sjoin(target_for_join, source_for_join, how='left', predicate='intersects')
joined_gdf = joined_gdf.drop_duplicates(subset=['__target_id'], keep='first')
joined_gdf[source_id_col] = joined_gdf[source_id_col].fillna('__NO_SOURCE__')
joined_gdf[source_orig_pop_col] = joined_gdf[source_orig_pop_col].fillna(0)
joined_gdf[source_pop_col] = joined_gdf[source_pop_col].fillna(0) # Float column

# 5. Apply LRM Manually
print("Applying LRM...")
# Assumes _integerize_lrm function is defined in the script
integer_results = joined_gdf.groupby(source_id_col).apply(
    lambda grp: _integerize_lrm(grp, source_pop_col, source_orig_pop_col)
)
integer_results = integer_results.reset_index(level=0, drop=True)
joined_gdf[output_col_name_int] = integer_results
joined_gdf[output_col_name_int] = joined_gdf[output_col_name_int].fillna(0).astype(int)

# 6. Prepare Final Output and Save
print("Saving results...")
final_gdf = target_data.merge(
    joined_gdf[[output_col_name_int]], left_index=True, right_index=True, how='left'
)
final_gdf[output_col_name_int] = final_gdf[output_col_name_int].fillna(0).astype(int)
os.makedirs(output_dir, exist_ok=True)
output_filepath_shp = os.path.join(output_dir, output_filename_shp)
# Select/rename columns as needed for shapefile before saving
cols_to_save = list(target_data.columns);
if 'geometry' in cols_to_save: cols_to_save.remove('geometry')
cols_to_save.extend([output_col_name_int, 'geometry'])
final_gdf[cols_to_save].to_file(output_filepath_shp, driver="ESRI Shapefile")

print("Done.")

Custom Function Usage (Currently Not Recommended)

The dasy_utils.py file contains the weighted_dasymetric_flexible function which offers more weighting options but currently fails due to the geopandas.overlay issue described above. If that issue were resolved (e.g., by updating libraries or identifying specific data problems), its usage would look like this:
Python

from dasy_utils import weighted_dasymetric_flexible

# Assume source_data, target_data are loaded and preprocessed (CRS, geometry fixing)
# Assume output_dir, output_filename_shp are defined

result_gdf = weighted_dasymetric_flexible(
    source_gdf=source_data,
    target_gdf=target_data,
    source_attribute="tot_teen",
    weight_attributes=None, # Or ['FLOORS'], etc.
    ancillary_vector_layers=None, # Or [land_use_gdf]
    ancillary_raster_path=None, # Or "path/to/density.tif"
    use_intersect_area_weight=True,
    binary_mask_layer=None, # Or mask_gdf
    mask_attribute=None, # Or "MASK_TYPE"
    mask_values_indicating_exclusion=[1],
    target_output_attribute="teen_est", # <= 10 chars for save_output
    integerize=True, # Use built-in LRM
    save_output=True,
    output_dir=output_dir,
    output_filename=output_filename_shp
)

print(result_gdf.head())

Files

    dasy_utils.py: Contains the weighted_dasymetric_flexible function and the _integerize_lrm helper function. Note: Relies on geopandas.overlay which currently fails with the test data.
    barebones_example_tobler_manual_lrm.py: Recommended example script demonstrating successful interpolation using tobler and manual LRM for integer results.
    (Other example scripts): May show different steps in the debugging process or alternative approaches.
