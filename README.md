# general_scripts
General Python scripts, mainly revolving around GDAL-based raster processing.

# Runtime Environment
Many of these scripts rely on external libraries like numpy, scipy, and (especially) GDAL. We _highly_ recommend you use Annaconda to create a Python 3.x virtual environment that handles package management. The GDAL libraries in particular can be a bit tricky to get working on your own; conda makes this really quite simple. We recommend using the GDAL libraries from the `conda-forge` channel instead of the default channel; they generally seem to be more up-to-date and have more options compiled in (though see https://github.com/conda-forge/gdal-feedstock/issues/219).

Most scripts were written in a Python 3.x environment. They should be usuable in Python 2.7 wihtout too much hassle (generally a `from __future__ import print_function` should suffice).

### A Note About GDAL and ECWs
We have not seen ECW read support included in the `conda-forge` GDAL libraries. If you need to read in ECWs, the easiest way we've found is to install QGIS, which comes with it's own python installation that has GDAL and the read-only ECW libaries included. Direct calls to these binaries (eg, `c:\Program Files\QGIS 2.18\bin\gdal_translate my_raster.ecw my_raster.tif`) should allow you to read/convert ECWs.

# Usage
Most of these scripts are written to be run from the command line without any arguments; relevant input is set directly as variables within the script. We may add argument parsers as time allows.

### Preaching To The Choir
When you're working with raster files, 87.835% of your problems can be cleared up by correctly managing your projection and NoData values. If your projection is in meters but your elevation is in feet, remember to use the proper conversion factor or change one of them to match the other. Make sure you've got a valid NoData value set, as the scripts here expect it to exist (and make sense).

# Script Descriptions

### raster_chunk_processing.py
RCP runs DEM smoothing and Kennelly and Steward's skyshading technique (https://gistbok.ucgis.org/bok-topics/terrain-representation) in parallel on arbitrarily-large rasters by dividing them into chunks and processing them individually. Currently implemented smoothing processes included a moving average blur (ie, Focal Statistics-Mean), gaussian blur, a blur developed by Mike Toews (https://gis.stackexchange.com/questions/9431/what-raster-smoothing-generalization-tools-are-available), and a call to Sun et al's mesh denoise program (http://www.cs.cf.ac.uk/meshfiltering/index_files/Page342.htm). Also included is a TPI calculator (which is really just a high-pass mean filter), a basic hillshade algorithm, and a CLAHE contrast stretcher (https://imagej.net/Enhance_Local_Contrast_(CLAHE)).

### rectified_mosaic.py
In the process of georectifying several hundered old aerial photos, it became clear that manually handling the overlaps and merging of the imagery would be impossible. This script automatically tiles the images into user-defined sized chunks, then builds a list of chunks to merge based on their distance to the center of their source image and the number of nodata pixels in the chunk. By feeding this list into gdalbuildvrt and then gdal_translate, you get a single, relativley seamless raster covering the entire subject area without image collars or other marginallia (assuming there is enough overlap in the original photos). The resulting raster can then have pyramids added and used in the desktop or web GIS of your choice.

### unknown_contours.py
A script for creating terrain joyplots and displaying the output with matplotlib. The resulting plot can be saved as a raster or vector, allowing further work in Illustrator/Inkskape. It's currently a little rough, but the basics are there, including arbitrary heading values (rotating the lines in relation to north) and adjusting the vertical exageration. 

### gdal_sky_model_2.py
A standalone, GDAL-based implementation of Kennelly and Steward's skyshading technique. Rolled into RCP, which can dramatically increase performance through parallelization.
