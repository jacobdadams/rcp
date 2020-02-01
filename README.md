# rcp
parallelized raster chunk processing

# Runtime Environment
TODO: Update 
Many of these scripts rely on external libraries like numpy, scipy, and (especially) GDAL. We _highly_ recommend you use Annaconda to create a Python 3.x virtual environment that handles package management. The GDAL libraries in particular can be a bit tricky to get working on your own; conda makes this really quite simple. We recommend using the GDAL libraries from the `conda-forge` channel instead of the default channel; they generally seem to be more up-to-date and have more options compiled in (though see https://github.com/conda-forge/gdal-feedstock/issues/219).

### Preaching To The Choir
When you're working with raster files, 87.835% of your problems can be cleared up by correctly managing your projection and NoData values. If your projection is in meters but your elevation is in feet, remember to use the proper conversion factor or change one of them to match the other. Make sure you've got a valid NoData value set, as the scripts here expect it to exist (and make sense).

# raster_chunk_processing.py
RCP runs DEM smoothing and Kennelly and Steward's skyshading technique (https://gistbok.ucgis.org/bok-topics/terrain-representation) in parallel on arbitrarily-large rasters by dividing them into chunks and processing them individually. Currently implemented smoothing processes included a moving average blur (ie, Focal Statistics-Mean), gaussian blur, a blur developed by Mike Toews (https://gis.stackexchange.com/questions/9431/what-raster-smoothing-generalization-tools-are-available), and a call to Sun et al's mesh denoise program (http://www.cs.cf.ac.uk/meshfiltering/index_files/Page342.htm). Also included is a TPI calculator (which is really just a high-pass mean filter), a basic hillshade algorithm, and a CLAHE contrast stretcher (https://imagej.net/Enhance_Local_Contrast_(CLAHE)).
