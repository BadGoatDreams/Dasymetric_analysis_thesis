# Python Pipeline for Tobler-based Dasymetric Mapping

Files: https://drive.google.com/drive/folders/1Ss_DURNRwxYdbQuYM99rnNtoDtI-5OyB?usp=drive_link

Mostly made with Gemini. This repository contains Python scripts to perform sum-preserving, integer-based dasymetric mapping, primarily using the `tobler` library for spatial interpolation and custom helper functions for data preparation and integerization. The main goal is disaggregating count data (e.g., population) from larger source zones to smaller target zones (e.g., building footprints).

## Description

Dasymetric mapping refines the spatial distribution of data using ancillary information. This project implements a pipeline approach that leverages the robust area-weighted interpolation capabilities of the `tobler` library and applies a manual Largest Remainder Method (LRM) to ensure the resulting allocated counts are integers and conserve the original totals within each source zone.

An earlier version attempted a custom implementation (`weighted_dasymetric_flexible` in `dasy_utils.py`) but encountered persistent issues with `geopandas.overlay`. **The `tobler`-based pipeline described here is the recommended and currently working approach.**

## Features (Pipeline)

* **Area-Weighted Interpolation:** Uses `tobler.area_weighted.area_interpolate` for reliable calculation of proportional values based on spatial overlap.
* **Sum-Preserving Integerization:** Implements the Largest Remainder Method (LRM) via helper functions to convert interpolated float values into integers while ensuring the sum matches the original source zone totals.
* **Data Preparation Helpers:** Includes functions to load data, handle CRS assignment and reprojection, and attempt geometry fixing using `buffer(0)`.
* **Modular Structure:** The logic is separated into a utility library (`dasy_pipeline_utils.py`) and a runner script (`run_dasy_pipeline.py`) for easier configuration and execution.
* **Shapefile Output:** Saves the final integerized results to an ESRI Shapefile.

## Current Status & Limitations

* **Working Approach:** The pipeline using `tobler` and manual LRM (orchestrated by `run_dasy_pipeline.py` using `dasy_pipeline_utils.py`) is functional and produces sum-preserving integer results based on area weighting.
* **Original Custom Function (`dasy_utils.py`):** Contains the `weighted_dasymetric_flexible` function offering more weighting options (rasters, attributes, masking). However, it is **currently non-functional** for the specific test data due to persistent failures in its core `geopandas.overlay` step, which returns empty intersections despite visual overlap and valid geometries. This file is kept for reference but is **not recommended for use** until the underlying `overlay` issues are resolved.
* **Weighting Limited:** The current working pipeline primarily demonstrates *area-weighted* interpolation via `tobler`. Extending it to use other weights (like building floors, land use factors) would require modifying the pipeline, potentially by calculating weights beforehand and passing them to more advanced `tobler` functions or implementing custom weighting logic around `tobler.area_weighted.area_interpolate`.

## Requirements

* Python 3.x
* GeoPandas
* Pandas
* NumPy
* Tobler
* Shapely (usually installed with GeoPandas)
* Math (standard library)
* OS (standard library)
* Warnings (standard library)

You can install the core requirements using pip:

```bash
pip install geopandas pandas numpy tobler

Usage

The recommended way to use this code is via the runner script:

    Configure run_dasy_pipeline.py:
        Open the run_dasy_pipeline.py script.
        Modify the variables in the # --- Configuration Parameters --- section:
            Set BASE_DATA_DIR, SOURCE_FILE, TARGET_FILE to your actual file paths.
            Set OUTPUT_DIR and OUTPUT_FILENAME for the desired output location and Shapefile name.
            Verify SOURCE_POP_COL (the population/count column in your source data) and SOURCE_ID_COL (a unique ID column for your source zones, like 'GEOID') are correct.
            Set OUTPUT_INT_COL to the desired name for the final integer result column (max 10 characters for Shapefile).
            Confirm SOURCE_ORIGINAL_CRS, TARGET_ORIGINAL_CRS, and TARGET_PROJECTED_CRS match your data.
    Place Files: Ensure run_dasy_pipeline.py and dasy_pipeline_utils.py are in the same directory (or that dasy_pipeline_utils.py is in your Python path).
    Run from Terminal:
    Bash

    python run_dasy_pipeline.py

    Output: The script will print progress messages and save the resulting Shapefile (containing the original target features plus the new integer population column) to the specified output directory. It also prints a comparison of the original total population and the final estimated integer total.

Core Functions (dasy_pipeline_utils.py)

This file contains the building blocks used by the run_dasy_pipeline.py script:

    _integerize_lrm(group_df, float_col, source_total_col):
        Internal helper function. Applies the Largest Remainder Method to a pandas DataFrame group representing features within a single source zone. Converts float estimates (float_col) to integers, ensuring their sum matches the group's original source total (source_total_col).
    load_and_prepare_data(filepath, target_crs, expected_original_crs=None, fix_geom=True, layer_name="Data"):
        Loads a vector file into a GeoDataFrame.
        Assigns expected_original_crs if the loaded data is missing CRS information.
        Reprojects the data to the specified target_crs.
        Optionally attempts to fix invalid geometries using .buffer(0) and removes empty geometries (fix_geom=True).
        Returns the prepared GeoDataFrame.
    run_area_interpolate_tobler(source_gdf, target_gdf, extensive_variables, source_id_col):
        Takes prepared source and target GeoDataFrames.
        Ensures specified extensive_variables (like population columns) exist and are numeric in the source.
        Calls tobler.area_weighted.area_interpolate to perform the core spatial weighting.
        Returns the target GeoDataFrame augmented with new columns containing the interpolated float results, along with the original total sum of the first extensive variable.
    link_source_info_for_lrm(interpolated_gdf, source_gdf, source_id_col, source_pop_col):
        Takes the float results from run_area_interpolate_tobler and the original prepared source data.
        Performs a spatial join (sjoin) to link each target feature back to its corresponding original source zone based on intersection.
        Adds the unique source ID (source_id_col) and the original source population value (source_pop_col, renamed internally) to the interpolated dataframe. This is necessary for grouping during LRM.
        Returns the joined GeoDataFrame.
    apply_lrm_and_finalize(joined_gdf, target_orig_gdf, float_col, source_id_col, source_orig_pop_col, output_col_name_int):
        Takes the joined_gdf (output from link_source_info_for_lrm).
        Groups the data by the linked source_id_col.
        Applies the _integerize_lrm helper function to each group to calculate sum-preserving integer results.
        Merges the final integer results (output_col_name_int) back onto the original target data structure (target_orig_gdf) using the index.
        Returns the final GeoDataFrame ready for saving.
    save_result_gdf(gdf, output_dir, output_filename, columns_to_save=None):
        Saves the provided GeoDataFrame (gdf) to an ESRI Shapefile.
        Creates the output_dir if it doesn't exist.
        Constructs the full output path.
        Optionally saves only a subset of columns specified in columns_to_save.
        Includes warnings/checks for Shapefile column name length limits.

File Structure

.
├── dasy_pipeline_utils.py   # Library with helper functions (uses Tobler)
├── run_dasy_pipeline.py     # Main script to configure and run the pipeline
├── dasy_utils.py            # Original custom function (DEPRECATED due to overlay issues)
├── README.md                # This file
└── data/                    # (Example directory - store your input data here)
    └── source/
    └── target/
└── output/                  # (Example directory - output files saved here)
