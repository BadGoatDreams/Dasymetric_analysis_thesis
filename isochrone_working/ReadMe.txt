 generate_isochrones_from_points

This R function generates isochrone polygons around points from a shapefile, using the Open Source Routing Machine (OSRM) API. It outputs these isochrones as a new shapefile, with each polygon containing the ID of the originating point.

## Function Description

The `generate_isochrones_from_points` function takes a point shapefile as input, calculates isochrones (areas reachable within specified time intervals) for each point, and exports the results to a new shapefile. It utilizes the `osrmIsochrone` function from the `osrm` R package to perform the isochrone calculations.

## Function Signature

```R
generate_isochrones_from_points(points_shapefile, output_shapefile, breaks = seq(0, 12, 2), osrm_server = "http://localhost:5000/", osrm_profile = "driving")

Arguments

    points_shapefile: (character) The file path to the input point shapefile.
    output_shapefile: (character) The file path where the output isochrone shapefile will be saved.
    breaks: (numeric vector, optional) A vector of time intervals (in minutes) that define the isochrone boundaries. Defaults to seq(0, 12, 2), which generates isochrones for 0-2, 2-4, 4-6, 6-8, 8-10, and 10-12 minute intervals.
    osrm_server: (character, optional) The URL of the OSRM server. Defaults to "http://localhost:5000/". Ensure your local OSRM server is running at this address or change it to the address of your server.
    osrm_profile: (character, optional) The OSRM routing profile (e.g., "driving", "walking", "cycling"). Defaults to "driving".

Return Value

The function returns a list containing:

    points: The original input point shapefile as an sf object.
    isochrones: The generated isochrone polygons as an sf object.

The isochrones are also exported to the specified output_shapefile.
Dependencies

    sf
    osrm

Example Usage
Code snippet

library(sf)
library(osrm)

# Replace with your actual file paths
points_file <- "your_points.shp"
output_file <- "your_isochrones.shp"

# Generate isochrones with default breaks and driving profile
results <- generate_isochrones_from_points(points_file, output_file)

# Or, generate isochrones with custom breaks and walking profile
# results <- generate_isochrones_from_points(points_file, output_file, breaks = c(5, 10, 15), osrm_profile = "walking")

# Access the results (optional)
points_data <- results$points
isochrones_data <- results$isochrones

#View the results (optional)
if(!is.null(isochrones_data)){
    plot(isochrones_data["isomax"])
}

# The isochrones are also saved to "your_isochrones.shp"

Setup

    Install R and RStudio (Recommended): If you haven't already, install R and RStudio.

    Install Required Packages: Run the following in your R console:
    Code snippet

    install.packages(c("sf", "osrm"))

    OSRM Server:
        Ensure you have a running OSRM server. If you don't have one, you can set up a local server.
        If using a local server, ensure it is running at the address specified by the osrm_server argument (default: "http://localhost:5000/").

    Input Shapefile:
        Replace "your_points.shp" with the actual path to your point shapefile.
        Ensure your input shapefile has an id column, as this is used to relate the isochrones back to the original points.

    Output Shapefile:
        Replace "your_isochrones.shp" with the desired path and filename for the output isochrone shapefile.

Notes

    The function assumes your OSRM server is configured correctly and has the necessary routing data for your area.
    The breaks argument allows you to control the time intervals for the isochrones, enabling you to analyze reachability at different time scales.
    If the input points shapefile doesn't have an id column, the isochrones will still be generated, but they won't have the originating point's ID.
    The delete_layer = TRUE argument in the st_write() function overwrites existing files with the same name.