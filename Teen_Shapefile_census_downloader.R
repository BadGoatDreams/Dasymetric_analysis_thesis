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


# Download census tract data for Oregon with the specified variables
teen_data <- get_acs(
  geography = "block group",
  variables = c(teen_population_15_17_m = "B01001_006",
                teen_population_18_19_m = "B01001_007",
                teen_population_15_17_f = "B01001_030",
                teen_population_18_19_f = "B01001_031"),  
  state = "OR",
  year = 2020,
  survey = "acs5"
)
# Clean up the column names
colnames(teen_data) <- gsub("E$", "", colnames(teen_data))

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
  left_join(teen_data, by = c("GEOID" = "GEOID"))

# Export the merged data as a shapefile
st_write(oregon_merged, "oregon_census_tracts.shp", driver = "ESRI Shapefile")
