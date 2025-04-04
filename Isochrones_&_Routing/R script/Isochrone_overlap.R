library(sf)
library(dplyr)

# Load isochrones and coffee shop points
isochrones <- st_read("5min_isochrones.shp") %>%
  filter(!st_is_empty(geometry)) %>%  # Remove empty geometries
  st_make_valid()                      # Ensure valid geometries

coffee_shops <- st_read("coffee_shops_points.shp") %>%
  filter(!st_is_empty(geometry)) %>%  # Remove empty geometries
  st_make_valid()

# Extract only valid polygons and points
isochrones <- st_collection_extract(isochrones, "POLYGON")
coffee_shops <- st_collection_extract(coffee_shops, "POINT")

# Ensure same CRS
isochrones <- st_transform(isochrones, st_crs(coffee_shops))

# Find isochrones overlapping coffee shops
isochrones$has_coffee <- lengths(st_intersects(isochrones, coffee_shops)) > 0

# Find overlapping isochrones
isochrones$overlaps <- lengths(st_overlaps(isochrones, isochrones)) > 1

# Classify
isochrones$class <- case_when(
  isochrones$overlaps & isochrones$has_coffee & lengths(st_intersects(isochrones, coffee_shops)) > 1 ~ 1,
  isochrones$overlaps & isochrones$has_coffee ~ 2,
  !isochrones$overlaps & isochrones$has_coffee & lengths(st_intersects(isochrones, coffee_shops)) > 1 ~ 3,
  !isochrones$overlaps & isochrones$has_coffee ~ 4,
  TRUE ~ 5
)

# Merge polygons by class
isochrones_dissolved <- isochrones %>%
  group_by(class) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

# Save output
st_write(isochrones_dissolved, "classified_isochrones.shp", delete_dsn = TRUE)
