# File: run_dasy_pipeline.py
# Description: Runs the dasymetric mapping pipeline using functions from dasy_pipeline_utils.py

import geopandas as gpd
import pandas as pd
import os
import sys
import traceback
import numpy as np

# --- Try to import the utility functions ---
try:
    import dasy_pipeline_utils as utils
except ImportError:
    print("ERROR: Could not import dasy_pipeline_utils.py.")
    print("Ensure dasy_pipeline_utils.py is in the same directory or your PYTHONPATH.")
    sys.exit(1) # Exit if utils can't be imported

# --- Configuration Parameters ---

# 1. File Paths
BASE_DATA_DIR = "C:/Users/plato/Documents/Dasymetric_analysis_thesis/dasy_test_gemini/"
SOURCE_FILE = os.path.join(BASE_DATA_DIR, "Teen_pop_benton/benton_census_tracts_teen_pop_short.shp")
TARGET_FILE = os.path.join(BASE_DATA_DIR, "residences_perhaps/residences_perhaps.shp")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "output_pipeline") # Changed output dir name slightly
OUTPUT_FILENAME = "teen_tobler_lrm_pipeline.shp" # Changed output name

# 2. Column Names
SOURCE_POP_COL = "tot_teen"   # Population column in source data
SOURCE_ID_COL = "GEOID"       # Unique ID column in source data
# Desired final integer column name (<= 10 characters for Shapefile)
OUTPUT_INT_COL = "teen_lrm"

# 3. Coordinate Reference Systems (CRS)
# Define the known/expected original geographic CRSs
SOURCE_ORIGINAL_CRS = "EPSG:4269" # NAD83 Geographic
TARGET_ORIGINAL_CRS = "EPSG:4326" # WGS84 Geographic
# Define the target projected CRS for calculations
TARGET_PROJECTED_CRS = "EPSG:2991"   # Oregon State Plane North

# --- Main Pipeline Execution ---
if __name__ == "__main__":

    print("=== Starting Dasymetric Pipeline ===")
    final_result_gdf = None # Initialize

    try:
        # --- Step 1 & 2: Load and Prepare Source and Target Data ---
        source_prepared_gdf = utils.load_and_prepare_data(
            filepath=SOURCE_FILE,
            target_crs=TARGET_PROJECTED_CRS,
            expected_original_crs=SOURCE_ORIGINAL_CRS,
            fix_geom=True, # Apply buffer(0) fix
            layer_name="Source"
        )

        target_prepared_gdf = utils.load_and_prepare_data(
            filepath=TARGET_FILE,
            target_crs=TARGET_PROJECTED_CRS,
            expected_original_crs=TARGET_ORIGINAL_CRS,
            fix_geom=True, # Apply buffer(0) fix
            layer_name="Target"
        )

        # --- Step 3: Run Tobler Area Interpolation ---
        interpolated_float_gdf, original_total = utils.run_area_interpolate_tobler(
            source_gdf=source_prepared_gdf,
            target_gdf=target_prepared_gdf,
            extensive_variables=[SOURCE_POP_COL],
            source_id_col=SOURCE_ID_COL
        )

        # --- Step 4: Link Source Info for LRM ---
        # Note: interpolated_float_gdf contains original target columns + the new float SOURCE_POP_COL
        linked_gdf = utils.link_source_info_for_lrm(
            interpolated_gdf=interpolated_float_gdf,
            source_gdf=source_prepared_gdf, # Pass the prepared source GDF again
            source_id_col=SOURCE_ID_COL,
            source_pop_col=SOURCE_POP_COL # Name of the float column AND the original source column
        )

        # --- Step 5: Apply LRM and Finalize ---
        # Define the name of the original source total column added by link_source_info
        source_orig_pop_col_linked = f"__{SOURCE_POP_COL}_orig"

        final_result_gdf = utils.apply_lrm_and_finalize(
            joined_gdf=linked_gdf,
            target_orig_gdf=target_prepared_gdf, # Use the prepared target GDF for final structure
            float_col=SOURCE_POP_COL,            # Name of float col from interpolation
            source_id_col=SOURCE_ID_COL,
            source_orig_pop_col=source_orig_pop_col_linked, # Name of linked original total col
            output_col_name_int=OUTPUT_INT_COL
        )

        # --- Step 6: Save Final Result ---
        # Define which columns from the original target data to keep, plus the new integer result
        cols_to_save = list(target_prepared_gdf.columns) # Start with original prepared target cols
        if 'geometry' in cols_to_save: cols_to_save.remove('geometry')
        # Add the NEW integer column
        cols_to_save.append(OUTPUT_INT_COL)

        utils.save_result_gdf(
            gdf=final_result_gdf,
            output_dir=OUTPUT_DIR,
            output_filename=OUTPUT_FILENAME,
            columns_to_save=cols_to_save # Pass the specific columns list
        )

        # --- Step 7: Final Verification Printout ---
        print("\n--- Final Results Summary ---")
        print("\nResult Head (Integerized):")
        print(final_result_gdf[cols_to_save].head()) # Print selected columns head

        estimated_total = final_result_gdf[OUTPUT_INT_COL].sum()
        print(f"\nOriginal Source Total: {original_total:.2f}") # Use total from run_area_interpolate
        print(f"LRM Est. Total:      {estimated_total}")
        print(f"Difference:          {original_total - estimated_total:.4f}") # Show difference with precision
        if np.isclose(original_total, estimated_total): print("Sum Conservation: OK")
        else: print("Sum Conservation: Check LRM Warnings")

        print("\n=== Pipeline Finished Successfully ===")

    # --- Error Handling ---
    except FileNotFoundError as e: print(f"\nERROR: Input file not found. Details: {e}")
    except ValueError as e: print(f"\nERROR: Input data issue. Details: {e}")
    except ImportError as e: print(f"\nERROR: Missing library. Details: {e}")
    except Exception as e:
         print(f"\n--- An unexpected error occurred ---")
         print(f"Error Type: {type(e).__name__}")
         print(f"Error Details: {e}")
         print("\nTraceback:")
         print(traceback.format_exc())
         print("\n--- Pipeline Terminated Due to Error ---")
         sys.exit(1) # Exit with error code