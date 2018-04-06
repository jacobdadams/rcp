# general_scripts
General Python scripts, mainly revolving around GDAL-based raster processing.

# Runtime Environment
Many of these scripts rely on external libraries like numpy, scipy, and (especially) GDAL. We _highly_ recommend you use Annaconda to create a Python 3.x virtual environment that handles package management. The GDAL libraries in particular can be a bit tricky to get working on your own; conda makes this really quite simple. We recommend using the GDAL libraries from the `conda-forge` channel instead of the default channel; they generally seem to be more up-to-date and have more options compiled in.

Most scripts were written in a Python 3.x environment. They should be usuable in Python 2.7 wihtout too much hassle (generally a `from __future__ import print_function` should suffice).

### A Note About GDAL and ECWs
We have not seen ECW read support included in the `conda-forge` GDAL libraries. If you need to read in ECWs, the easiest way we've found is to install QGIS, which comes with it's own python installation that has GDAL and the read-only ECW libaries included. Direct calls to these binaries (eg, `c:\Program Files\QGIS 2.18\bin\gdal_translate my_raster.ecw my_raster.tif`) should allow you to read/convert ECWs.

# Usage
Most of these scripts are written to be run from the command line without any arguments; relevant input is set directly as variables within the script. We may add argument parsers as time allows.

### Preacing To The Choir
When you're working with raster files, 87.835% of your problems can be cleared up by correctly managing your projection and NoData values. If your projection is in meters but your elevation is in feet, remember to use the proper conversion factor or change one of them to match the other. Make sure you've got a valid NoData value set, as the scripts here expect it to exist (and make sense).
