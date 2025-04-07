# Install and load required packages
if (!require(tidycensus)) install.packages("tidycensus")
if (!require(dplyr)) install.packages("dplyr")
if (!require(sf)) install.packages("sf")

library(tidycensus)
library(dplyr)
library(sf)

# Set your Census API key
census_api_key("54ac47162c6e58ddd931d877cd791f9dae6b4f81", install = TRUE)

# Define the variables for housing and income data
variables <- c(
  median_household_income = "B19013_001",
  median_home_value = "B25077_001",
  total_population = "B01003_001",
  total_housing_units = "B25001_001",
  occupied_housing_units = "B25002_002",
  vacant_housing_units = "B25002_003"
)

# Download census tract data for Oregon with the specified variables
oregon_data <- get_acs(
  geography = "tract",
  variables = variables,
  state = "OR",
  year = 2020,
  survey = "acs5",
  output = "wide"
)

# Clean up the column names
colnames(oregon_data) <- gsub("E$", "", colnames(oregon_data))

# Download the spatial data for census tracts in Oregon
oregon_geo <- get_acs(
  geography = "tract",
  variables = "B01003_001",
  state = "OR",
  year = 2020,
  survey = "acs5",
  geometry = TRUE
)

# Join the downloaded data with the spatial data
oregon_merged <- oregon_geo %>%
  select(GEOID, NAME, geometry) %>%
  left_join(oregon_data, by = c("GEOID" = "GEOID"))

# Export the merged data as a shapefile
st_write(oregon_merged, "oregon_census_tracts.shp", driver = "ESRI Shapefile")
