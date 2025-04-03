# File: dasy_pipeline_utils.py
# Description: Utility functions for a dasymetric mapping pipeline using Tobler.

import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import os
import math

# --- Attempt to import Tobler ---
try:
    from tobler.area_weighted import area_interpolate
except ImportError:
    warnings.warn("Tobler library not found. Install with 'pip install tobler'. Area interpolation function will fail.")
    # Define a dummy function to allow script loading but fail at runtime if called
    def area_interpolate(*args, **kwargs):
        raise ImportError("Tobler library not found or could not be imported.")

# --- Helper Function for Largest Remainder Method ---
def _integerize_lrm(group_df: pd.DataFrame, float_col: str, source_total_col: str) -> pd.Series:
    """
    Applies Largest Remainder Method within a source zone group.
    Internal helper function.
    """
    # Get the single source total for this group (should be the same for all rows)
    try:
        source_total = group_df[source_total_col].iloc[0]
    except IndexError:
         warnings.warn(f"LRM: Group {group_df.name} appears empty or lacks source total. Returning zeros.")
         return pd.Series(0, index=group_df.index, dtype=int)

    if pd.isna(source_total) or source_total == 0: # Handle NaN or zero source total
        return pd.Series(0, index=group_df.index, dtype=int)

    # Ensure float_col is numeric
    group_df[float_col] = pd.to_numeric(group_df[float_col], errors='coerce').fillna(0.0)

    # Calculate floor and remainder
    floors = group_df[float_col].apply(lambda x: math.floor(x) if pd.notna(x) else 0)
    remainders = group_df[float_col] - floors

    # Calculate how many units to distribute
    total_floored = floors.sum()
    to_distribute = int(round(source_total - total_floored))

    if to_distribute < 0:
        warnings.warn(f"LRM: Negative difference ({to_distribute}) for group {group_df.name}. Capping distribution at 0.")
        to_distribute = 0
    # Check if distribution is needed (handle potential floating point comparison issues)
    elif to_distribute == 0 and np.isclose(source_total, total_floored):
        return floors.astype(int)

    # Get indices sorted by remainder (descending), handle ties consistently
    sorted_indices = remainders.sort_values(ascending=False).index

    # Distribute the difference
    final_integers = floors.copy().astype(int) # Start with integer floors
    count = 0
    for idx in sorted_indices:
        if count >= to_distribute:
            break
        if idx in final_integers.index: # Ensure index exists
            final_integers.loc[idx] += 1
            count += 1
        else:
             warnings.warn(f"LRM: Index {idx} from sorted remainders not found in final_integers for group {group_df.name}. Skipping.")


    # Final check
    final_sum = final_integers.sum()
    if not np.isclose(final_sum, source_total):
         warnings.warn(f"LRM integer sum ({final_sum}) differs slightly from source total ({source_total}) for group {group_df.name}. Check float precision.")

    return final_integers


# --- Main Pipeline Functions ---

def load_and_prepare_data(
    filepath: str,
    target_crs: str,
    expected_original_crs: str = None,
    fix_geom: bool = True,
    layer_name: str = "Data" # Generic name for messages
) -> gpd.GeoDataFrame:
    """
    Loads vector data, assigns original CRS if missing, reprojects, and optionally fixes geometries.

    Args:
        filepath (str): Path to the vector file (Shapefile, GeoPackage, etc.).
        target_crs (str): The target projected CRS (e.g., "EPSG:2991").
        expected_original_crs (str, optional): The expected geographic CRS if the file lacks one
                                               (e.g., "EPSG:4269", "EPSG:4326"). Defaults to None.
        fix_geom (bool): If True, apply buffer(0) and remove empty geometries. Defaults to True.
        layer_name (str): Name to use for logging messages (e.g., "Source", "Target"). Defaults to "Data".

    Returns:
        gpd.GeoDataFrame: The prepared GeoDataFrame.

    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If CRS is missing and expected_original_crs is not provided, or if data is empty after cleaning.
        Exception: Reraises exceptions from geopandas/fiona during file reading.
    """
    print(f"--- Loading and Preparing {layer_name} Data ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{layer_name} file not found: {filepath}")

    print(f"Loading: {filepath}")
    gdf = gpd.read_file(filepath)

    # Assign original CRS if missing
    if gdf.crs is None:
        if expected_original_crs:
            print(f"{layer_name} data missing CRS. Assuming {expected_original_crs} and assigning...")
            gdf = gdf.set_crs(expected_original_crs, allow_override=True)
        else:
            raise ValueError(f"{layer_name} data missing CRS and no 'expected_original_crs' provided.")
    elif expected_original_crs and gdf.crs != expected_original_crs and gdf.crs != target_crs:
         # Check if CRS is different from both expected original and target
         warnings.warn(f"{layer_name} data has CRS '{gdf.crs}', differs from expected '{expected_original_crs}'. Proceeding with reprojection.")

    # Reproject if necessary
    if gdf.crs != target_crs:
        print(f"Reprojecting {layer_name} data from {gdf.crs} to {target_crs}...")
        gdf = gdf.to_crs(target_crs)
    else:
        print(f"{layer_name} data already in target CRS ({target_crs}).")

    # Fix geometries if requested
    if fix_geom:
        print(f"Attempting to fix {layer_name} geometries using buffer(0)...")
        invalid_before = gdf[~gdf.geometry.is_valid].shape[0]
        gdf['geometry'] = gdf.geometry.buffer(0)
        invalid_after = gdf[~gdf.geometry.is_valid].shape[0]
        print(f"{layer_name} invalid geometries: Before={invalid_before}, After={invalid_after}")

        initial_rows = len(gdf)
        gdf = gdf[~gdf.geometry.is_empty]
        removed_empty = initial_rows - len(gdf)
        if removed_empty > 0:
            print(f"Removed {removed_empty} empty geometries from {layer_name} data.")
        print(f"{layer_name} rows after cleaning: {len(gdf)}")

    if gdf.empty:
        raise ValueError(f"{layer_name} dataset became empty after loading/cleaning. Cannot proceed.")

    print(f"{layer_name} data prepared successfully.")
    return gdf


def run_area_interpolate_tobler(
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    extensive_variables: list,
    source_id_col: str # Needed for checking data integrity
) -> tuple[gpd.GeoDataFrame, float]:
    """
    Runs Tobler's area interpolation and returns float results.

    Args:
        source_gdf (gpd.GeoDataFrame): Prepared source GeoDataFrame (Projected CRS).
                                       Must contain geometry, extensive_variables, and source_id_col.
        target_gdf (gpd.GeoDataFrame): Prepared target GeoDataFrame (Projected CRS).
        extensive_variables (list): List of column names in source_gdf to interpolate (e.g., ['population']).
        source_id_col (str): Name of the unique ID column in source_gdf.

    Returns:
        tuple[gpd.GeoDataFrame, float]:
            - GeoDataFrame with target geometries and new columns for interpolated float values.
            - Overall sum of the first extensive variable from the source data.
    """
    print(f"\n--- Running Tobler Area Interpolation for: {extensive_variables} ---")

    # Ensure ID and geometry columns exist
    if source_id_col not in source_gdf.columns:
        raise ValueError(f"Source ID column '{source_id_col}' not found in source_gdf.")
    if 'geometry' not in source_gdf.columns or 'geometry' not in target_gdf.columns:
        raise ValueError("Geometry column missing from source or target GeoDataFrame.")

    # Ensure extensive variables are numeric and store original total
    original_total_overall = 0
    for var in extensive_variables:
        if var not in source_gdf.columns:
            raise ValueError(f"Extensive variable '{var}' not found in source_gdf.")
        if not pd.api.types.is_numeric_dtype(source_gdf[var]):
            warnings.warn(f"Converting source column '{var}' to numeric (filling NA with 0).")
            source_gdf[var] = pd.to_numeric(source_gdf[var], errors='coerce').fillna(0)
        if var == extensive_variables[0]: # Store total of the first variable
            original_total_overall = source_gdf[var].sum()
            print(f"Original Source Total '{var}': {original_total_overall}")

    # Keep only necessary columns from source for interpolation
    source_cols_needed = [source_id_col, 'geometry'] + extensive_variables
    source_subset = source_gdf[source_cols_needed]

    # Perform interpolation
    interpolated_gdf = area_interpolate(
        source_df=source_subset,
        target_df=target_gdf, # Pass full target GDF to retain its columns
        extensive_variables=extensive_variables
    )

    # Check results
    for var in extensive_variables:
        if var not in interpolated_gdf.columns:
            raise RuntimeError(f"Column '{var}' was not added by area_interpolate.")
        # Fill any NaNs in results with 0 (can happen if target didn't overlap any source)
        interpolated_gdf[var] = interpolated_gdf[var].fillna(0.0)
        print(f"Area interpolation complete for '{var}'. Float results added.")
        print(f"DEBUG: Sum of float results for '{var}': {interpolated_gdf[var].sum()}")

    return interpolated_gdf, original_total_overall


def link_source_info_for_lrm(
    interpolated_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    source_id_col: str,
    source_pop_col: str # The specific column used for interpolation total
) -> gpd.GeoDataFrame:
    """
    Spatially joins interpolated target data with original source data
    to link source IDs and original totals needed for LRM.
    """
    print("\n--- Re-linking Source Information for LRM ---")

    # Ensure required columns exist
    if source_pop_col not in interpolated_gdf.columns:
        raise ValueError(f"Float result column '{source_pop_col}' not in interpolated_gdf.")
    if source_id_col not in source_gdf.columns:
        raise ValueError(f"Source ID column '{source_id_col}' not in source_gdf.")
    if source_pop_col not in source_gdf.columns:
        raise ValueError(f"Original population column '{source_pop_col}' not in source_gdf.")

    # Prepare target side for join (geometry + float result + unique ID)
    target_for_join = interpolated_gdf[[source_pop_col, 'geometry']].copy()
    # Create a reliable unique ID if target doesn't have one already
    if '__target_id_join' not in target_for_join.columns:
        target_for_join['__target_id_join'] = range(len(target_for_join))
        target_id_col_join = '__target_id_join'
    else: # If user had an ID, assume it's unique
         target_id_col_join = '__target_id_join' # Or use user's ID column? Let's stick to temp one

    # Prepare source side for join (geometry + source ID + original source total)
    source_orig_pop_col = f"__{source_pop_col}_orig" # Temp name for original total
    source_for_join = source_gdf[[source_id_col, source_pop_col, 'geometry']].rename(
        columns={source_pop_col: source_orig_pop_col}
    )

    # Perform spatial join
    print("Performing spatial join (target 'intersects' source)...")
    joined_gdf = gpd.sjoin(target_for_join, source_for_join, how='left', predicate='intersects')

    # Handle potential multiple matches (keep first match per target feature)
    initial_len = len(joined_gdf)
    joined_gdf = joined_gdf.drop_duplicates(subset=[target_id_col_join], keep='first')
    if len(joined_gdf) < initial_len:
        warnings.warn(f"Removed {initial_len - len(joined_gdf)} duplicate matches from sjoin result.")

    # Check required columns are present
    if source_id_col not in joined_gdf.columns or source_orig_pop_col not in joined_gdf.columns:
         raise RuntimeError(f"Required source columns ('{source_id_col}', '{source_orig_pop_col}') lost during sjoin.")

    # Fill NaNs for targets that didn't overlap any source
    # Assign a unique placeholder ID and zero population
    no_source_mask = joined_gdf[source_id_col].isnull()
    if no_source_mask.any():
        print(f"Found {no_source_mask.sum()} target features with no overlapping source. Assigning zero pop.")
        joined_gdf.loc[no_source_mask, source_id_col] = '__NO_SOURCE__' + joined_gdf.loc[no_source_mask, target_id_col_join].astype(str)
        joined_gdf.loc[no_source_mask, source_orig_pop_col] = 0
        joined_gdf.loc[no_source_mask, source_pop_col] = 0 # Ensure float result is also 0

    # Ensure float column has no NaNs (should have been handled by area_interpolate already)
    joined_gdf[source_pop_col] = joined_gdf[source_pop_col].fillna(0.0)

    print("Source information linked.")
    return joined_gdf


def apply_lrm_and_finalize(
    joined_gdf: gpd.GeoDataFrame,
    target_orig_gdf: gpd.GeoDataFrame, # The original target GDF (pre-interpolation)
    float_col: str,
    source_id_col: str,
    source_orig_pop_col: str, # Name of the added column holding original source totals
    output_col_name_int: str
) -> gpd.GeoDataFrame:
    """
    Applies LRM via groupby and merges integer results back to the original target GDF structure.
    """
    print(f"\n--- Applying LRM and Finalizing Output ---")

    # Apply LRM using the helper function via groupby
    print(f"Applying LRM grouped by '{source_id_col}'...")
    try:
        # Ensure the grouping column is suitable type (string/object often safest)
        grouping_col_type = joined_gdf[source_id_col].dtype
        if not pd.api.types.is_string_dtype(grouping_col_type) and not pd.api.types.is_object_dtype(grouping_col_type):
             warnings.warn(f"Converting source ID column '{source_id_col}' to string for robust grouping.")
             joined_gdf[source_id_col] = joined_gdf[source_id_col].astype(str)

        # Apply LRM
        integer_results = joined_gdf.groupby(source_id_col, group_keys=False).apply(
            lambda grp: _integerize_lrm(grp, float_col, source_orig_pop_col)
        )
        # Check result type and align index
        if isinstance(integer_results, pd.DataFrame): # If apply returns df instead of series
            if len(integer_results.columns) == 1:
                 integer_results = integer_results.iloc[:, 0]
            else: raise TypeError("LRM apply result has unexpected shape.")
        # Ensure index matches joined_gdf
        integer_results = integer_results.reindex(joined_gdf.index).fillna(0)

    except Exception as e:
        warnings.warn(f"LRM Groupby/Apply failed: {e}. Cannot create integer column.")
        raise # Re-raise the error to stop execution

    # Assign integer results to the joined dataframe
    joined_gdf[output_col_name_int] = integer_results.astype(int)
    print(f"Integerization complete. Column '{output_col_name_int}' created.")
    print(f"DEBUG: Sum of LRM integer results in joined_gdf: {joined_gdf[output_col_name_int].sum()}")


    # Merge final integer column back to the original target geodataframe structure
    print("Merging integer results back to original target structure...")
    # Use index from original target_gdf
    final_gdf = target_orig_gdf.merge(
        joined_gdf[[output_col_name_int]], # Select only the final integer column and index
        left_index=True,
        right_index=True,
        how='left' # Keep all original targets
    )
    # Fill any original targets that didn't get a result with 0
    final_gdf[output_col_name_int] = final_gdf[output_col_name_int].fillna(0).astype(int)

    print("Final GeoDataFrame prepared.")
    return final_gdf


def save_result_gdf(
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    output_filename: str,
    columns_to_save: list = None
):
    """
    Saves the resulting GeoDataFrame to an ESRI Shapefile.

    Args:
        gdf (gpd.GeoDataFrame): The final GeoDataFrame to save.
        output_dir (str): Directory to save the file.
        output_filename (str): Filename (e.g., 'output.shp').
        columns_to_save (list, optional): Specific columns to include in the output.
                                          If None, saves all columns. Defaults to None.
    """
    print(f"\n--- Saving Output ---")
    if not output_filename.lower().endswith(".shp"):
        warnings.warn(f"Output filename '{output_filename}' does not end with .shp. Saving as ESRI Shapefile anyway.")

    # Check column names if saving specific columns
    final_columns = gdf.columns
    if columns_to_save:
        # Ensure geometry is included if not explicitly listed
        geom_col = gdf.geometry.name
        if geom_col not in columns_to_save:
            columns_to_save.append(geom_col)
        # Check for missing columns
        missing = [col for col in columns_to_save if col not in final_columns]
        if missing: raise ValueError(f"Columns specified in 'columns_to_save' not found in GDF: {missing}")
        # Check length of specified columns
        long_names = [col for col in columns_to_save if len(col) > 10]
        if long_names: warnings.warn(f"Columns names > 10 chars in 'columns_to_save', may be truncated by Shapefile driver: {long_names}")
        gdf_to_save = gdf[columns_to_save]
    else:
        # Check length of all columns if saving all
        long_names = [col for col in final_columns if len(col) > 10]
        if long_names: warnings.warn(f"Columns names > 10 chars found, may be truncated by Shapefile driver: {long_names}")
        gdf_to_save = gdf

    # Create directory and save
    try:
        os.makedirs(output_dir, exist_ok=True)
        full_output_path = os.path.join(output_dir, output_filename)
        print(f"Saving file to: {full_output_path}")
        gdf_to_save.to_file(full_output_path, driver="ESRI Shapefile")
        print(f"Successfully saved output to {full_output_path}")
    except Exception as e:
        raise IOError(f"Failed to save output Shapefile to '{full_output_path}'. Error: {e}")