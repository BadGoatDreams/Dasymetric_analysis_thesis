# barebones_example_tobler_manual_lrm.py

import geopandas as gpd
import pandas as pd
import warnings
import os
import math
import numpy as np

# --- Import Tobler Function ---
try:
    from tobler.area_weighted import area_interpolate
except ImportError:
    print("ERROR: Could not import tobler. Make sure it is installed (`pip install tobler`).")
    exit()

# --- LRM Helper Function (Copied from dasy_utils.py) ---
# NOTE: Included directly in script since the main function isn't used
def _integerize_lrm(group_df: pd.DataFrame, float_col: str, source_total_col: str) -> pd.Series:
    """Applies Largest Remainder Method within a source zone group."""
    # Get the single source total for this group (should be the same for all rows)
    source_total = group_df[source_total_col].iloc[0]

    if source_total == 0: # No population to distribute
        return pd.Series(0, index=group_df.index, dtype=int)

    # Ensure float_col is numeric
    group_df[float_col] = pd.to_numeric(group_df[float_col], errors='coerce').fillna(0.0)

    # Calculate floor and remainder
    floors = group_df[float_col].apply(math.floor)
    remainders = group_df[float_col] - floors

    # Calculate how many units to distribute
    total_floored = floors.sum()
    # Round difference to nearest int - crucial for handling potential float inaccuracies
    to_distribute = int(round(source_total - total_floored))

    if to_distribute < 0:
        warnings.warn(f"LRM: Negative difference ({to_distribute}) for group {group_df.name}. Capping distribution at 0.")
        to_distribute = 0
    elif to_distribute == 0 and np.allclose(source_total, total_floored):
         # If exactly zero diff and totals match, return floors
        return floors.astype(int)
    # If diff is zero but totals don't match due to float precision, may still need to distribute below

    # Get indices sorted by remainder (descending), handle ties consistently (keep='first')
    # Add a small random element to break ties randomly if needed, but usually sort is stable
    # group_df['_rem_'] = remainders
    # group_df['_rand_'] = np.random.rand(len(group_df))
    # sorted_indices = group_df.sort_values(by=['_rem_', '_rand_'], ascending=[False, False]).index
    # group_df = group_df.drop(columns=['_rem_', '_rand_']) # cleanup temp columns
    sorted_indices = remainders.sort_values(ascending=False).index


    # Distribute the difference
    final_integers = floors.copy()
    # Add 1 to the top 'to_distribute' rows based on remainder
    count = 0
    for idx in sorted_indices:
        if count >= to_distribute:
            break
        final_integers.loc[idx] += 1
        count += 1

    # Final check - LRM should guarantee sum preservation if implemented correctly
    final_sum = final_integers.sum()
    if not np.isclose(final_sum, source_total):
         # If still off, likely float precision issue earlier. Adjust last element? Risky.
         warnings.warn(f"LRM integer sum ({final_sum}) differs from source total ({source_total}) for group {group_df.name}. Check input float precision.")

    return final_integers.astype(int)

# --- 1. Define File Paths and Columns ---
base_data_dir = "C:/Users/plato/Documents/Dasymetric_analysis_thesis/dasy_test_gemini/"
source_filepath = os.path.join(base_data_dir, "Teen_pop_benton/benton_census_tracts_teen_pop_short.shp")
target_filepath = os.path.join(base_data_dir, "residences_perhaps/residences_perhaps.shp")
output_dir = os.path.join(base_data_dir, "output_shapefiles")
output_filename_shp = "teen_tobler_lrm_integer.shp" # New output name

# --- Columns ---
source_pop_col = "tot_teen"   # Source population column
source_id_col = "GEOID"       # Unique ID column in source data
# Define the desired final integer column name (<= 10 chars)
output_col_name_int = "teen_lrm"

# --- 3. Load Data ---
try:
    print(f"Loading source data: {source_filepath}")
    source_data = gpd.read_file(source_filepath)
    print(f"Loading target data: {target_filepath}")
    target_data = gpd.read_file(target_filepath)

    # --- Check for Source ID Column ---
    if source_id_col not in source_data.columns:
        raise ValueError(f"Source ID column '{source_id_col}' not found in {source_filepath}.")

    # --- 4. CRS Definition and Reprojection ---
    source_original_crs = "EPSG:4269"; target_original_crs = "EPSG:4326"; target_crs = "EPSG:2991"
    if source_data.crs is None: source_data = source_data.set_crs(source_original_crs, allow_override=True)
    if target_data.crs is None: target_data = target_data.set_crs(target_original_crs, allow_override=True)
    print(f"Ensuring both layers are in target CRS: {target_crs}")
    if source_data.crs != target_crs: source_data = source_data.to_crs(target_crs)
    if target_data.crs != target_crs: target_data = target_data.to_crs(target_crs)

    # --- Optional: Fix Geometries using buffer(0) ---
    print("Attempting to fix potential invalid geometries using buffer(0)...")
    source_data['geometry'] = source_data.geometry.buffer(0)
    target_data['geometry'] = target_data.geometry.buffer(0)
    source_data = source_data[~source_data.geometry.is_empty]
    target_data = target_data[~target_data.geometry.is_empty]
    print(f"Data cleaned. Source rows: {len(source_data)}, Target rows: {len(target_data)}")
    if source_data.empty or target_data.empty: raise ValueError("Empty dataset after cleaning.")

    # --- Ensure population column is numeric ---
    if not pd.api.types.is_numeric_dtype(source_data[source_pop_col]):
        print(f"Converting source column '{source_pop_col}' to numeric...")
        source_data[source_pop_col] = pd.to_numeric(source_data[source_pop_col], errors='coerce').fillna(0)
    original_total_overall = source_data[source_pop_col].sum() # Store overall total
    print(f"Original Source Total '{source_pop_col}': {original_total_overall}")

    # --- 5. Call Tobler Area Interpolation (Produces Float Results) ---
    print(f"\nRunning Tobler area interpolation for variable: '{source_pop_col}'...")
    # This adds a column named 'tot_teen' to a copy of target_data
    interpolated_float_gdf = area_interpolate(
        source_df=source_data[[source_id_col, source_pop_col, 'geometry']], # Keep only needed cols
        target_df=target_data, # Pass original target df (or necessary cols + geom)
        extensive_variables=[source_pop_col]
    )
    # Check if the output column exists
    if source_pop_col not in interpolated_float_gdf.columns:
        raise ValueError(f"Column '{source_pop_col}' not found after area_interpolate.")
    print(f"Area interpolation complete. Float results in column '{source_pop_col}'.")
    print("\nHead of Float Results:")
    print(interpolated_float_gdf[[source_pop_col]].head())
    print(f"DEBUG: Sum of float results: {interpolated_float_gdf[source_pop_col].sum()}")

    # --- 6. Prepare for Manual LRM: Re-link Source Info ---
    print("\nRe-linking source information for integerization...")
    # We need to know which original source zone each target building belongs to *mostly*.
    # Use sjoin with 'within' or 'intersects' (within might be safer if buildings don't cross boundaries after buffer)
    # Keep relevant columns: target unique id (use index if none), float result, source id, original source total
    target_for_join = interpolated_float_gdf[[source_pop_col, 'geometry']].copy()
    target_for_join['__target_id'] = target_for_join.index # Use index as unique target ID

    source_for_join = source_data[[source_id_col, source_pop_col, 'geometry']].copy()
    # Rename source population column to avoid clash with interpolated one
    source_orig_pop_col = f"__{source_pop_col}_orig"
    source_for_join = source_for_join.rename(columns={source_pop_col: source_orig_pop_col})

    # Perform the spatial join
    # Use 'intersects' as 'within' might fail if buffer(0) changed things slightly
    # Using 'left' join to keep all target buildings, even if they somehow don't join (shouldn't happen)
    joined_gdf = gpd.sjoin(target_for_join, source_for_join, how='left', predicate='intersects', lsuffix='target', rsuffix='source')

    # Handle potential multiple matches if a target intersects multiple sources (less likely after interpolation)
    # Keep the first match if duplicates occur based on target ID
    joined_gdf = joined_gdf.drop_duplicates(subset=['__target_id'], keep='first')

    # Check if the required columns are present after join
    if source_id_col not in joined_gdf.columns:
        raise ValueError(f"Source ID column '{source_id_col}' lost during spatial join.")
    if source_orig_pop_col not in joined_gdf.columns:
         raise ValueError(f"Original source population column '{source_orig_pop_col}' lost during spatial join.")
    if source_pop_col not in joined_gdf.columns: # This is the float column
         raise ValueError(f"Float result column '{source_pop_col}' lost during spatial join.")

    # Fill NaN for source ID/total if any target didn't overlap (assign them 0 pop)
    joined_gdf[source_id_col] = joined_gdf[source_id_col].fillna('__NO_SOURCE__')
    joined_gdf[source_orig_pop_col] = joined_gdf[source_orig_pop_col].fillna(0)
    joined_gdf[source_pop_col] = joined_gdf[source_pop_col].fillna(0) # Ensure float col has no NaNs


    # --- 7. Apply Manual LRM using Groupby ---
    print(f"\nApplying Largest Remainder Method using group by '{source_id_col}'...")

    # Use the helper function defined above
    # Pass the name of the float column and the original source total column
    integer_results = joined_gdf.groupby(source_id_col).apply(
        lambda grp: _integerize_lrm(grp, source_pop_col, source_orig_pop_col)
    )

    # The result might be multi-level or need re-alignment. Flatten if needed.
    # If groupby().apply() returns Series for each group, concat/stack might be needed
    # Or does it return a single series aligned with the original df index? Let's assume it does for now.
    # We need to merge this back based on index.
    integer_results = integer_results.reset_index(level=0, drop=True) # Drop the group key index level if present
    joined_gdf[output_col_name_int] = integer_results # Assign LRM results

    # Fill any NaNs possibly created during LRM/merge with 0
    joined_gdf[output_col_name_int] = joined_gdf[output_col_name_int].fillna(0).astype(int)

    print(f"Integerization complete. Integer results in column '{output_col_name_int}'.")
    print(f"DEBUG: Sum of LRM integer results: {joined_gdf[output_col_name_int].sum()}")
    # Compare overall sums (should match)
    print(f"DEBUG: Original overall source sum: {original_total_overall}")


    # --- 8. Prepare Final Output & Save ---
    # Select necessary columns from the original target df and merge the result
    final_gdf = target_data.merge(
        joined_gdf[[output_col_name_int]], # Select only the final integer column and index
        left_index=True,
        right_index=True,
        how='left' # Keep all original targets
    )
    # Fill any target that didn't get a result (e.g., removed empty) with 0
    final_gdf[output_col_name_int] = final_gdf[output_col_name_int].fillna(0).astype(int)


    # --- Save Results as Shapefile ---
    os.makedirs(output_dir, exist_ok=True)
    output_filepath_shp = os.path.join(output_dir, output_filename_shp)
    print(f"\nSaving integerized results to Shapefile: {output_filepath_shp}")

    save_col_name = output_col_name_int
    if len(save_col_name) > 10: raise ValueError(f"Final column name '{save_col_name}' too long.")

    # Select columns to save
    cols_to_save = list(target_data.columns) # Start with original target cols
    if 'geometry' in cols_to_save: cols_to_save.remove('geometry')
    cols_to_save.append(save_col_name)
    cols_to_save.append('geometry')

    missing_cols = [col for col in cols_to_save if col not in final_gdf.columns]
    if missing_cols: raise ValueError(f"Columns missing before saving: {missing_cols}")

    final_gdf[cols_to_save].to_file(output_filepath_shp, driver="ESRI Shapefile")
    print(f"Successfully saved to {output_filepath_shp}")

    print("\nTobler + Manual LRM integerization example finished.")
    print("\nResult Head (Integerized):")
    print(final_gdf[cols_to_save[:-1]].head())

    # Final Sum Check
    estimated_total = final_gdf[save_col_name].sum()
    print(f"\nOriginal Source Total: {original_total_overall}")
    print(f"LRM Est. Total:      {estimated_total}")
    print(f"Difference:          {original_total_overall - estimated_total}")


except FileNotFoundError as e: print(f"\nERROR: Input file not found. Details: {e}")
except ValueError as e: print(f"\nERROR: Input data issue. Details: {e}")
except ImportError as e: print(f"\nERROR: Missing library (maybe tobler?). Details: {e}")
except Exception as e:
     import traceback
     print(f"\nAn unexpected error occurred: {e}")
     print("Traceback:"); print(traceback.format_exc())