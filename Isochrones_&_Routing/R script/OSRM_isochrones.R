library(sf) # Load the sf package for spatial data handling
library(osrm) # Load the osrm package for OSRM routing functions

generate_isochrones_from_points <- function(points_shapefile, # Input: Path to the points shapefile
                                            output_shapefile, # Input: Path to the output isochrone shapefile
                                            breaks = seq(0, 15, 5), # Input: Time breaks for isochrones (in minutes) first number is the starting value, second is ending, third is interval. (0,10,5) would produce 2 isochrones of 5 and 10 minutes respectively.
                                            osrm_server = "http://localhost:5000/", # Input: OSRM server URL
                                            osrm_profile = "driving") { # Input: OSRM routing profile (e.g., "driving", "walking")
  
  options(osrm.server = osrm_server) # Set the OSRM server URL for the current R session
  options(osrm.profile = osrm_profile) # Set the OSRM routing profile for the current R session
  
  points_sf <- st_read(points_shapefile, quiet = TRUE) # Read the input points shapefile into an sf object
  
  all_isochrones <- list() # Initialize an empty list to store the generated isochrones
  
  for (i in 1:nrow(points_sf)) { # Loop through each point in the input shapefile
    current_point <- points_sf[i, ] # Get the current point as an sf object
    
    tryCatch({ # Use tryCatch to handle potential errors during isochrone generation
      isochrones <- osrmIsochrone(loc = current_point, breaks = breaks) # Generate isochrones for the current point
      isochrones$point_id <- points_sf$id[i] # Add the point ID from the input to the isochrones
      all_isochrones[[i]] <- isochrones # Store the generated isochrones in the list
    }, error = function(e) { # Handle errors if isochrone generation fails
      warning(paste("Error generating isochrones for point", i, ":", e$message)) # Display a warning message
      all_isochrones[[i]] <- NULL # Store NULL in the list if isochrone generation failed
    })
  }
  
  combined_isochrones <- do.call(rbind, all_isochrones[!sapply(all_isochrones, is.null)]) # Combine all generated isochrones into a single sf object, removing NULL entries
  
  if(!is.null(combined_isochrones)){ # Check if any isochrones were generated
    st_write(combined_isochrones, output_shapefile, delete_layer = TRUE) # Write the isochrones to the output shapefile, overwriting existing files
    print(paste("Isochrones exported to:", output_shapefile)) # Print a message indicating the output file path
  }
  
  return(list(points = points_sf, isochrones = combined_isochrones)) # Return the input points and generated isochrones as a list
}

# Example usage (replace with your actual file paths):
points <- "C:/Users/plato/Desktop/Dasymetric_analysis_thesis/adjusted_points_final/adjusted_points_final.shp"
output <- "people_points_output.shp"
testerisos <- generate_isochrones_from_points(points,output)
#st_write(testerisos$geometry, "test_output.shp")
