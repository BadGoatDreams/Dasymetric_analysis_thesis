# File: dasy_utils.py (Corrected Syntax v4)
# Requires: geopandas, pandas, numpy
# Optional (for raster support): rasterio, rasterstats

import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import os
import math # Needed for floor

# Optional imports for raster data handling
try:
    import rasterio
    import rasterstats
    RASTER_SUPPORT = True
except ImportError:
    RASTER_SUPPORT = False

# --- Helper Function for Largest Remainder Method ---
# (_integerize_lrm function remains the same)
def _integerize_lrm(group_df: pd.DataFrame, float_col: str, source_total: float) -> pd.Series:
    if source_total == 0: return pd.Series(0, index=group_df.index)
    group_df[float_col] = pd.to_numeric(group_df[float_col], errors='coerce').fillna(0.0)
    floors = group_df[float_col].apply(math.floor); remainders = group_df[float_col] - floors
    total_floored = floors.sum(); to_distribute = int(round(source_total - total_floored))
    if to_distribute < 0: warnings.warn(f"LRM: Negative diff ({to_distribute}). Cap 0."); to_distribute = 0
    elif to_distribute == 0: return floors.astype(int)
    sorted_indices = remainders.sort_values(ascending=False).index; final_integers = floors.copy()
    indices_to_increment = sorted_indices[:to_distribute]; final_integers.loc[indices_to_increment] += 1
    final_sum = final_integers.sum();
    if not np.isclose(final_sum, source_total): warnings.warn(f"LRM sum ({final_sum}) != source ({source_total}).")
    return final_integers.astype(int)

def weighted_dasymetric_flexible(
    source_gdf: gpd.GeoDataFrame, target_gdf: gpd.GeoDataFrame, source_attribute: str,
    weight_attributes: list = None, ancillary_vector_layers: list = None,
    ancillary_raster_path: str = None, raster_stats_method: str = 'mean',
    weight_multiplier: float = 1.0, default_weight_value: float = 1.0,
    final_default_weight: float = 0.0, use_intersect_area_weight: bool = True,
    binary_mask_layer = None, mask_attribute: str = None,
    mask_values_indicating_exclusion: list = [1], target_output_attribute: str = None,
    crs_enforcement: str = 'warn', integerize: bool = False,
    save_output: bool = False, output_dir: str = None, output_filename: str = None
) -> gpd.GeoDataFrame:
    """ (Docstring remains the same) """
    # --- 1. Input Validation & Setup ---
    # (Section 1 includes previous fix for default name)
    print("Starting..."); print("Validating...")
    if not isinstance(source_gdf, gpd.GeoDataFrame): raise TypeError("source_gdf must be GDF")
    if not isinstance(target_gdf, gpd.GeoDataFrame): raise TypeError("target_gdf must be GDF")
    if source_attribute not in source_gdf.columns: raise ValueError(f"Source attr '{source_attribute}' missing.")
    if target_output_attribute is None:
        target_output_attribute = f"est_{source_attribute}";
        if len(target_output_attribute) > 10: target_output_attribute = target_output_attribute[:10]
        print(f"Output column defaulted: {target_output_attribute}")
    if save_output:
        if not output_dir or not output_filename: raise ValueError("output_dir/filename required if save_output=True.")
        if len(target_output_attribute) > 10: raise ValueError(f"Output column '{target_output_attribute}' > 10 chars.")
        if not output_filename.lower().endswith(".shp"): warnings.warn(f"Filename '{output_filename}' != .shp.")
    if weight_attributes is None: weight_attributes = []
    source = source_gdf.copy(); target = target_gdf.copy()
    if '__source_id' not in source.columns: source['__source_id'] = range(len(source))
    if '__target_id' not in target.columns: target['__target_id'] = range(len(target))

    # --- 2. Load Ancillary/Mask Layers & CRS Checks ---
    # (Section 2 includes previous fixes)
    print("Loading layers & checking CRS...")
    layers_to_load = {'ancillary': ancillary_vector_layers if ancillary_vector_layers else [], 'mask': [binary_mask_layer] if binary_mask_layer is not None else []}
    loaded_ancillary = []; loaded_mask = None
    def check_crs(layer_gdf, layer_name, base_crs):
        if not isinstance(layer_gdf, gpd.GeoDataFrame): warnings.warn(f"{layer_name} not GDF."); return False, base_crs
        if layer_gdf.crs is None: print(f"{layer_name} missing CRS."); return False, base_crs
        current_base_crs = base_crs; ok=True
        if current_base_crs is None: current_base_crs = layer_gdf.crs
        if layer_gdf.crs != current_base_crs: message = f"{layer_name} CRS ({layer_gdf.crs}) != base ({current_base_crs})."; ok=False
        if ok and not layer_gdf.crs.is_projected: warnings.warn(f"CRS ({layer_gdf.crs}) for {layer_name} not projected.")
        if not ok:
            if crs_enforcement == 'error': raise ValueError(message)
            elif crs_enforcement == 'warn': warnings.warn(message)
        return ok, current_base_crs
    base_crs = None; crs_ok, base_crs = check_crs(source, 'Source', base_crs); crs_ok_target, base_crs = check_crs(target, 'Target', base_crs)
    if not crs_ok_target and crs_enforcement == 'error': raise ValueError("Target CRS check failed.")
    for i, item in enumerate(layers_to_load['ancillary']): # Load Ancillary
        anc_gdf = None; layer_name = f"Ancillary {i}";
        if isinstance(item, gpd.GeoDataFrame): anc_gdf = item.copy(); layer_name += " (GDF)"
        elif isinstance(item, str) and os.path.exists(item):
            layer_name += f" ({os.path.basename(item)})"
            try: anc_gdf = gpd.read_file(item)
            except Exception as e: raise ValueError(f"Load ancillary fail: {item}. Err: {e}")
        else: warnings.warn(f"Ancillary item {i} invalid. Skip."); continue
        if anc_gdf is not None: crs_ok, base_crs = check_crs(anc_gdf, layer_name, base_crs);
        if crs_ok or crs_enforcement != 'error': loaded_ancillary.append(anc_gdf)
    if layers_to_load['mask']: # Load Mask
        item = layers_to_load['mask'][0]; layer_name = "Mask"; mask_gdf_loaded_temp = None
        if isinstance(item, gpd.GeoDataFrame): mask_gdf_loaded_temp = item.copy(); layer_name += " (GDF)"
        elif isinstance(item, str) and os.path.exists(item):
             layer_name += f" ({os.path.basename(item)})"
             try: mask_gdf_loaded_temp = gpd.read_file(item)
             except Exception as e: raise ValueError(f"Load mask fail: {item}. Err: {e}")
        else: warnings.warn(f"Mask item invalid. Disable mask.")
        if mask_gdf_loaded_temp is not None:
            crs_ok, base_crs = check_crs(mask_gdf_loaded_temp, layer_name, base_crs)
            if crs_ok or crs_enforcement != 'error':
                if not mask_attribute: raise ValueError("Mask requires mask_attribute.");
                if mask_attribute not in mask_gdf_loaded_temp.columns: raise ValueError(f"Mask attr '{mask_attribute}' missing.")
                loaded_mask = mask_gdf_loaded_temp

    # --- 3. Enrich Target GDF ---
    # (Section 3 remains the same)
    print("Enriching target layer...")
    target_enriched = target.copy(); original_target_geom_name = target_enriched.geometry.name
    for i, anc_gdf in enumerate(loaded_ancillary):
        try:
            target_enriched = gpd.sjoin(target_enriched, anc_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix=f'_anc{i}')
            target_enriched = target_enriched.drop_duplicates(subset=['__target_id'], keep='first')
            if original_target_geom_name not in target_enriched.columns and target_enriched.geometry.name != original_target_geom_name: warnings.warn(f"Geom col name changed after sjoin {i}.")
        except Exception as e: warnings.warn(f"Sjoin ancillary fail {i}: {e}. Skip.") ; continue
    missing_attrs = [attr for attr in weight_attributes if attr not in target_enriched.columns]
    if missing_attrs: warnings.warn(f"Weight attrs missing: {missing_attrs}. Ignored."); weight_attributes = [a for a in weight_attributes if a in target_enriched.columns]

    # --- 4. Intersection ---
    print("Performing spatial intersection...")
    try:
        intersection = gpd.overlay(target_enriched.rename_geometry('__geom_left__'), source.rename_geometry('__geom_right__'), how='intersection', keep_geom_type=False)
        if target.geom_type.isin(['Polygon', 'MultiPolygon']).all():
             intersection = intersection[intersection.geometry.is_valid & ~intersection.geometry.is_empty & intersection.geometry.isin(['Polygon', 'MultiPolygon'])]
             intersection['__intersect_area'] = intersection.geometry.area
        elif target.geom_type.isin(['Point', 'MultiPoint']).all():
              intersection = intersection[intersection.geometry.is_valid & ~intersection.geometry.is_empty & intersection.geometry.isin(['Point', 'MultiPoint'])]
              intersection['__intersect_area'] = 1.0
        else:
             warnings.warn("Target geometry type mixed."); intersection = intersection[intersection.geometry.is_valid & ~intersection.geometry.is_empty]
             intersection['__intersect_area'] = intersection.geometry.apply(lambda g: g.area if g.geom_type in ['Polygon', 'MultiPolygon'] else 1.0)
        print(f"DEBUG: Intersect Area calculated. Min={intersection['__intersect_area'].min()}, Max={intersection['__intersect_area'].max()}, Sum={intersection['__intersect_area'].sum()}")
    except Exception as e: print(f"Debug: target cols: {target_enriched.columns}"); print(f"Debug: source cols: {source.columns}"); raise RuntimeError(f"Overlay error: {e}")

    # --- Check if intersection is empty ---
    if intersection.empty:
        warnings.warn("Intersection empty.")
        target[target_output_attribute] = 0.0 # Assign 0 to the output column
        target = target.drop(columns=['__target_id'], errors='ignore')

        # --- Corrected Saving Logic for Empty Case ---
        if save_output:
            print(f"Attempting to save empty result (with zeros) to Shapefile...")
            # Checks already done, proceed to try saving
            try:
                os.makedirs(output_dir, exist_ok=True)
                full_output_path = os.path.join(output_dir, output_filename)
                # Check output col name length again just before saving
                if len(target_output_attribute) > 10:
                    raise ValueError(f"Output col name '{target_output_attribute}' > 10 chars.")
                print(f"Saving file to: {full_output_path}")
                # Save the target dataframe which now has the zero column
                target.to_file(full_output_path, driver="ESRI Shapefile")
                print(f"Successfully saved empty result to {full_output_path}")
            except Exception as e:
                raise IOError(f"Failed to save empty output Shapefile. Error: {e}")
        # --- End Corrected Saving Logic ---
        return target # Return early

    # --- 5. Weight Calculation ---
    # (Section 5 remains the same)
    print("Calculating weights...")
    intersection['__weight'] = 1.0
    for attr in weight_attributes: # 5a. Vector Attributes
        if attr in intersection.columns: attr_values = intersection[attr]; attr_weight = pd.to_numeric(attr_values, errors='coerce').fillna(default_weight_value); intersection['__weight'] *= attr_weight
    if ancillary_raster_path: # 5b. Raster Weight
        if not RASTER_SUPPORT: raise ImportError("Raster libraries not found.")
        else: print(f"Extracting raster values ('{raster_stats_method}')...")
        try:
            valid_geom_mask = intersection.geometry.is_valid & ~intersection.geometry.is_empty; valid_geom_intersection = intersection[valid_geom_mask]
            if not valid_geom_intersection.empty:
                 stats = rasterstats.zonal_stats(valid_geom_intersection, ancillary_raster_path, stats=[raster_stats_method], geojson_out=False, nodata=np.nan, affine=None, all_touched=True)
                 raster_values = pd.Series([s[raster_stats_method] if s and s[raster_stats_method] is not None else np.nan for s in stats], index=valid_geom_intersection.index).fillna(default_weight_value)
                 intersection = intersection.merge(raster_values.rename('__raster_weight'), left_index=True, right_index=True, how='left')
                 intersection['__raster_weight'] = intersection['__raster_weight'].fillna(default_weight_value); intersection['__weight'] *= intersection['__raster_weight']; intersection = intersection.drop(columns=['__raster_weight'])
            else: warnings.warn("No valid geoms for raster stats.")
        except FileNotFoundError: raise ValueError(f"Raster file not found: {ancillary_raster_path}")
        except Exception as e: warnings.warn(f"Raster weight error: {e}. Ignored.")
    if use_intersect_area_weight and target.geom_type.isin(['Polygon', 'MultiPolygon']).all(): # 5c. Intersect Area
        if '__intersect_area' in intersection.columns: area_weight = intersection['__intersect_area'].fillna(0.0).clip(lower=0); intersection['__weight'] *= area_weight
        else: warnings.warn("__intersect_area missing. Skip area weight.")
    intersection['__weight'] *= weight_multiplier # 5d. Global Multiplier
    nan_weights_before_fill = intersection['__weight'].isnull().sum() # 5e. Final Fill/Clip
    if nan_weights_before_fill > 0: print(f"DEBUG: Found {nan_weights_before_fill} NaN weights before final fill.")
    intersection['__weight'] = intersection['__weight'].replace([np.inf, -np.inf], np.nan).fillna(final_default_weight).clip(lower=0)
    print(f"DEBUG: Weights calculated. Min={intersection['__weight'].min()}, Max={intersection['__weight'].max()}, Mean={intersection['__weight'].mean()}")

    # --- 6. Apply Binary Mask ---
    # (Section 6 remains the same)
    if loaded_mask is not None:
        print("Applying binary mask...")
        try:
            mask_geom_col = loaded_mask.geometry.name; intersection_geom_col = intersection.geometry.name
            intersection_geoms = intersection[[intersection_geom_col]].copy()
            masked_join = gpd.sjoin(intersection_geoms, loaded_mask[[mask_attribute, mask_geom_col]], how='inner', predicate='intersects', lsuffix='left', rsuffix='mask')
            mask_indices_to_exclude = masked_join[masked_join[mask_attribute].isin(mask_values_indicating_exclusion)].index.unique()
            if not mask_indices_to_exclude.empty: print(f"Masking out {len(mask_indices_to_exclude)} pieces."); intersection.loc[mask_indices_to_exclude, '__weight'] = final_default_weight
            else: print("No pieces overlapped mask exclusion zones.")
        except Exception as e: warnings.warn(f"Masking failed: {e}. Skipped.")

    # --- 7. Apportionment ---
    # (Section 7 remains the same - includes conditional integerization)
    print("Apportioning source attribute...")
    if source_attribute not in intersection.columns: possible_src_cols = [c for c in intersection.columns if source_attribute in c]; raise ValueError(f"Source attr '{source_attribute}' missing. Possible: {possible_src_cols}")
    source_weight_sum = intersection.groupby('__source_id')['__weight'].sum().rename('__total_source_weight')
    print(f"DEBUG: Source weight sums (raw):\n{source_weight_sum.head()}"); print(f"DEBUG: Source weight sums stats: Min={source_weight_sum.min()}, Max={source_weight_sum.max()}, NonZero={(source_weight_sum > 0).sum()}")
    intersection = intersection.merge(source_weight_sum, left_on='__source_id', right_index=True, how='left'); intersection['__total_source_weight'] = intersection['__total_source_weight'].fillna(0)
    source_data_map = source.set_index('__source_id')[source_attribute]
    density_map = (source_data_map / source_weight_sum).fillna(0).replace([np.inf, -np.inf], 0)
    print(f"DEBUG: Density map (raw):\n{density_map.head()}"); print(f"DEBUG: Density map stats: Min={density_map.min()}, Max={density_map.max()}, NonZero={(density_map > 0).sum()}")
    intersection['__density'] = intersection['__source_id'].map(density_map).fillna(0)
    float_result_col = f"__{target_output_attribute}_float"
    intersection[float_result_col] = intersection['__density'] * intersection['__weight']
    print(f"DEBUG: Calculated '{float_result_col}'. Min={intersection[float_result_col].min()}, Max={intersection[float_result_col].max()}, Sum={intersection[float_result_col].sum()}")
    if integerize:
        print("Applying Largest Remainder Method..."); intersection = intersection.merge(source_data_map.rename('__source_total_val'), left_on='__source_id', right_index=True, how='left')
        try:
            integer_results = intersection.groupby('__source_id', group_keys=False).apply(lambda grp: _integerize_lrm(grp, float_result_col, grp['__source_total_val'].iloc[0]))
            intersection[target_output_attribute] = integer_results; intersection = intersection.drop(columns=[float_result_col, '__source_total_val'], errors='ignore')
            print(f"DEBUG: Integerized '{target_output_attribute}'. Min/Max/Sum={intersection[target_output_attribute].min()}/{intersection[target_output_attribute].max()}/{intersection[target_output_attribute].sum()}")
        except Exception as e: warnings.warn(f"Integerization failed: {e}. Using float."); intersection[target_output_attribute] = intersection[float_result_col]; intersection = intersection.drop(columns=[float_result_col, '__source_total_val'], errors='ignore')
    else: intersection.rename(columns={float_result_col: target_output_attribute}, inplace=True)
    distributed_sum = intersection.groupby('__source_id')[target_output_attribute].sum(); original_sum = source_data_map.reindex(distributed_sum.index).fillna(0)
    sums_df = pd.DataFrame({'distributed': distributed_sum, 'original': original_sum})
    if integerize: warnings.warn("Correction factor skipped for integer results."); sums_df['correction_factor'] = 1.0
    else: sums_df['correction_factor'] = sums_df.apply(lambda row: row['original'] / row['distributed'] if row['distributed'] != 0 else 1.0, axis=1).fillna(1.0)
    print(f"DEBUG: Correction factors:\n{sums_df.head()}")
    mismatch = ~np.isclose(sums_df['distributed'], sums_df['original'])
    if not integerize and mismatch.any():
        print(f"DEBUG: Applying correction factor for {mismatch.sum()} source zones (float results).")
        intersection = intersection.merge(sums_df[['correction_factor']], left_on='__source_id', right_index=True, how='left')
        intersection[target_output_attribute] *= intersection['correction_factor'].fillna(1.0)
        if 'correction_factor' in intersection.columns: intersection = intersection.drop(columns=['correction_factor'])
    else: print("DEBUG: Correction factor not needed or skipped.")

    # --- 8. Aggregation ---
    # (Section 8 remains the same)
    print("Aggregating results...")
    final_target_values = intersection.groupby('__target_id')[target_output_attribute].sum()
    target = target.merge(final_target_values.rename(target_output_attribute), left_on='__target_id', right_index=True, how='left')
    if integerize: target[target_output_attribute] = target[target_output_attribute].fillna(0).astype(int)
    else: target[target_output_attribute] = target[target_output_attribute].fillna(0.0)

    # --- 9. Cleanup & Optional Save ---
    print("Cleaning up...")
    target = target.drop(columns=['__target_id'], errors='ignore')

    # --- Corrected Automatic Saving Logic ---
    if save_output:
        print(f"Attempting to save output to Shapefile...")
        # Checks done earlier, proceed to save
        try:
            os.makedirs(output_dir, exist_ok=True) # Ensure dir exists
            full_output_path = os.path.join(output_dir, output_filename)
            # Final check on col name length
            if len(target_output_attribute) > 10:
                 raise ValueError(f"Output column name '{target_output_attribute}' > 10 chars.")
            print(f"Saving file to: {full_output_path}")
            target.to_file(full_output_path, driver="ESRI Shapefile")
            print(f"Successfully saved output to {full_output_path}")
        except Exception as e:
            raise IOError(f"Failed to save output Shapefile to '{full_output_path}'. Error: {e}")
    # --- End Corrected Saving Logic ---

    print("Dasymetric mapping complete.")
    return target

# --- End of Function Definition ---