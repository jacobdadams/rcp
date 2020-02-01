'''
Methods that can be called by raster_chunk_processing.py
'''

import contextlib
import math
import numba
import os
import subprocess
import tempfile
import warnings
import multiprocessing as mp
import numpy as np
from astropy.convolution import convolve_fft
from osgeo import gdal

import settings


def WriteASC(in_array, asc_path, xll, yll, c_size, nodata=-37267):
    '''
    Writes an np.array to a .asc file, which is the most accessible format for
    mdenoise.exe.
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    asc_path:       The output path for the .asc file
    xll:            X coordinate for lower left corner; actual position is
                    irrelevant for mdenoise blur method below.
    y11:            Y coordinate for lower left corner; see above.
    c_size:         Square dimension of raster cell.
    nodata:         NoData value for .asc file.
    '''

    rows = in_array.shape[0]
    cols = in_array.shape[1]
    ncols = "ncols {}\n".format(cols)
    nrows = "nrows {}\n".format(rows)
    xllcorner = "xllcorner {}\n".format(xll)
    yllcorner = "yllcorner {}\n".format(yll)
    cellsize = "cellsize {}\n".format(c_size)
    nodata_value = "nodata_value {}\n".format(nodata)

    with open(asc_path, 'w') as f:
        # Write Header
        f.write(ncols)
        f.write(nrows)
        f.write(xllcorner)
        f.write(yllcorner)
        f.write(cellsize)
        f.write(nodata_value)

        # Write data
        for i in range(rows):
            row = " ".join("{0}".format(n) for n in in_array[i, :])
            f.write(row)
            f.write("\n")


def blur_mean(in_array, s_nodata, radius):
    '''
    Performs a simple blur based on the average of nearby values. Uses circular
    mask from Inigo Hernaez Corres, https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    This is the equivalent of ArcGIS' Focal Statistics (Mean) raster processing
    tool using a circular neighborhood.
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    radius:         The radius (in grid cells) of the circle used to define
                    nearby pixels. A larger value creates more pronounced
                    smoothing. The diameter of the circle becomes 2*radius + 1,
                    to account for the subject pixel.
    '''

    # Using modified circular mask from user Inigo Hernaez Corres, https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    # Using convolve_fft instead of gf(np.mean), which massively speeds up
    # execution (from ~3 hours to ~5 minutes on one dataset).
    nan_array = np.where(in_array == s_nodata, np.nan, in_array)
    diameter = 2 * radius + 1
    # Create a circular mask
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 > radius**2
    # Determine number of Falses (ie, cells in kernel not masked out)
    valid_entries = mask.size - np.count_nonzero(mask)
    # Create a kernel of 1/(the number of valid entries after masking)
    kernel = np.ones((diameter, diameter)) / (valid_entries)
    # Mask away the non-circular areas
    kernel[mask] = 0

    # kernel = [[4.5, 0, 0],
    #           [0, 0.001, 0],
    #           [0, 0, -5]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        circular_mean = convolve_fft(nan_array, kernel,
                                     nan_treatment='interpolate')#, normalize_kernel=False)

    return circular_mean


def blur_gauss(in_array, s_nodata, sigma, radius=30):
    '''
    Performs a gaussian blur on an array of elevations. Modified from Mike
    Toews, https://gis.stackexchange.com/questions/9431/what-raster-smoothing-generalization-tools-are-available
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    radius:         The radius (in grid cells) of the gaussian blur kernel
    '''

    # This comment block is old and left here for posterity
    # Change all NoData values to mean of valid values to fix issues with
    # massive (float32.max) NoData values completely overwhelming other array
    # data. Using mean instead of 0 gives a little bit more usable data on
    # edges.
    # Create masked array to get mean of valid data
    # masked_array = np.ma.masked_values(in_array, s_nodata)
    # array_mean = masked_array.mean()
    # # Create new array that will have NoData values replaced by array_mean
    # cleaned_array = np.copy(in_array)
    # np.putmask(cleaned_array, cleaned_array==s_nodata, array_mean)

    # convolving: output pixel is the sum of the multiplication of each value
    # covered by the kernel with the associated kernel value (the kernel is a
    # set size/shape and each position has a value, which is the multiplication
    # factor used in the convolution).

    # Create new array with s_nodata values set to np.nan (for edges of raster)
    nan_array = np.where(in_array == s_nodata, np.nan, in_array)

    # build kernel (Gaussian blur function)
    # g is a 2d gaussian distribution of size (2*size) + 1
    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    # Gaussian distribution
    twosig = 2 * sigma**2
    g = np.exp(-(x**2 / twosig + y**2 / twosig)) / (twosig * math.pi)
    #LoG
    #g = (-1/(math.pi*sigma**4))*(1-(x**2 + y**2)/twosig)*np.exp(-(x**2 / twosig + y**2 / twosig)) / (twosig)

    #g = 1 - g
    # Convolve the data and Gaussian function (do the Gaussian blur)
    # Supressing runtime warnings due to NaNs (they just get hidden by NoData
    # masks in the supper_array rebuild anyways)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Use the astropy function because fftconvolve does not like np.nan
        #smoothed = fftconvolve(padded_array, g, mode="valid")
        smoothed = convolve_fft(nan_array, g, nan_treatment='interpolate', normalize_kernel=False)
        # Uncomment the following line for a high-pass filter
        #smoothed = nan_array - smoothed

    return smoothed


def blur_toews(in_array, s_nodata, radius):
    '''
    Performs a blur on an array of elevations based on convolution kernel from
    Mike Toews, https://gis.stackexchange.com/questions/9431/what-raster-smoothing-generalization-tools-are-available
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    radius:         The radius (in grid cells) of the blur kernel
    '''

    # Create new array with s_nodata values set to np.nan (for edges of raster)
    nan_array = np.where(in_array == s_nodata, np.nan, in_array)

    # build kernel
    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x**2 / float(radius) + y**2 / float(radius)))
    g = (g / g.sum()).astype(nan_array.dtype)
    #g = 1 - g

    # Supressing runtime warnings due to NaNs (they just get hidden by NoData
    # masks in the supper_array rebuild anyways)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smoothed = convolve_fft(nan_array, g, nan_treatment='interpolate')
        # Uncomment the following line for a high-pass filter
        #smoothed = nan_array - smoothed
    return smoothed


def hillshade(in_array, az, alt, nodata, cell_size, scale=False):
    '''
    Custom implmentation of hillshading, using the algorithm from the source
    code for gdaldem. The inputs and outputs are the same as in gdal or ArcGIS.
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    az:             The sun's azimuth, in degrees.
    alt:            The sun's altitude, in degrees.
    nodata:         The source raster's nodata value.
    scale:          When true, stretches the result to 1-255. CAUTION: If using
                    as part of a parallel or multi-chunk process, each chunk
                    has different min and max values, which leads to different
                    stretching for each chunk.
    '''

    # Create new array wsith nodata values set to np.nan (for edges)
    nan_array = np.where(in_array == nodata, np.nan, in_array)

    # x = np.zeros(nan_array.shape)
    # y = np.zeros(nan_array.shape)

    # Conversion between mathematical and nautical azimuth
    az = 90. - az

    azrad = az * np.pi / 180.
    altrad = alt * np.pi / 180.

    x, y = np.gradient(nan_array, cell_size, cell_size, edge_order=2)
    # x, y = np.gradient(in_array, cell_size, cell_size, edge_order=2)

    sinalt = np.sin(altrad)
    cosaz = np.cos(azrad)
    cosalt = np.cos(altrad)
    sinaz = np.sin(azrad)
    xx_plus_yy = x * x + y * y
    alpha = y * cosaz * cosalt - x * sinaz * cosalt
    x = None
    y = None
    shaded = (sinalt - alpha) / np.sqrt(1 + xx_plus_yy)
    # result is +-1, scale to 0-255, mult by weight
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # newmax = 255
        # newmin = 0
        # oldmax = 1
        # oldmin = -1
        # ((newmax-newmin)(val-oldmin))/(oldmax-oldmin)+newmin
        result = 127.5 * (shaded + 1)

   # if scale:
    #     # Scale to 1-255 (stretches min value to 1, max to 255)
    #     # ((newmax-newmin)(val-oldmin))/(oldmax-oldmin)+newmin
    #     # Supressing runtime warnings due to NaNs (they just get hidden by
    #     # NoData masks in the supper_array rebuild anyways)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=RuntimeWarning)
    #         newmax = 255
    #         newmin = 1
    #         oldmax = np.nanmax(shaded255)
    #         oldmin = np.nanmin(shaded255)
    #
    #     result = (newmax-newmin) * (shaded255-oldmin) / (oldmax-oldmin) + newmin
    # else:
    #     result = shaded255
    #
    # return result

    return result


def skymodel(in_array, lum_lines, overlap, nodata, res):
    '''
    Creates a unique hillshade based on a skymodel, implmenting the method
    defined in Kennelly and Stewart (2014), A Uniform Sky Illumination Model to
    Enhance Shading of Terrain and Urban Areas.

    in_array:       The input array, should be read using the supper_array
                    technique from below.
    lum_lines:      The azimuth, altitude, and weight for each iteration of the
                    hillshade. Stored as an array lines, with each line being
                    an array of [az, alt, weight].
    '''

    # initialize skyshade as 0's
    skyshade = np.zeros(in_array.shape)

    # If it's all NoData, just return an array of 0's
    if in_array.mean() == nodata:
        return skyshade

    # Multiply elevation by 5 as per original paper
    in_array *= 5
    # nan_array *= 5

    # Loop through luminance file lines to calculate multiple hillshades
    for line in lum_lines:
        az = float(line[0])
        alt = float(line[1])
        weight = float(line[2])
        # Only pass a small overlapping in_array to hillshade- the shade
        # overlap is way larger than needed for the hillshade
        if overlap > 20:
            hs_overlap = overlap - 20
        else:
            hs_overlap = 0
        shade = np.zeros(in_array.shape)
        shade[hs_overlap:-hs_overlap, hs_overlap:-hs_overlap] = hillshade(
                    in_array[hs_overlap:-hs_overlap, hs_overlap:-hs_overlap],
                    az=az, alt=alt, nodata=nodata*5, cell_size=res, scale=False,
                    )
        shadowed = shadows(in_array, az, alt, res, overlap, nodata*5)
        # shade = hillshade(nan_array, az=az, alt=alt, scale=False) * weight
        # shadowed = shadowing.shadows(nan_array, az, alt, cell_size, overlap, nodata)
        # scale from 0-255 to 1-255, apply weight to scaled (I think arcpy hillshades range from 1-255, with 0 being nodata)
        # Now instead of shadowed areas always being 0, they'll be 1*scale- it will still contribute to final summed raster
        # ((newmax-newmin)(val-oldmin))/(oldmax-oldmin)+newmin
        # scaled = 0.996078431*(shade*shadowed) + 1
        # skyshade += (0.996078431*(shade*shadowed) + 1) * weight

        skyshade += shade * shadowed * weight

        shade = None

    return skyshade


@numba.jit(nopython=True)
def shadows(in_array, az, alt, res, overlap, nodata):
    # Rows = i = y values, cols = j = x values
    rows = in_array.shape[0]
    cols = in_array.shape[1]
    shadow_array = np.ones(in_array.shape)  # init to 1 (not shadowed), change to 0 if shadowed
    max_elev = np.max(in_array)

    az = 90. - az  # convert from 0 = north, cw to 0 = east, ccw

    azrad = az * np.pi / 180.
    altrad = alt * np.pi / 180.
    delta_j = math.cos(azrad)
    delta_i = -1. * math.sin(azrad)
    tanaltrad = math.tan(altrad)

    mult_size = 1
    max_steps = 600

    already_shadowed = 0

    # precompute idx distances
    distances = []
    for d in range(1, max_steps):
        distance = d * res
        step_height = distance * tanaltrad
        i_distance = delta_i * d
        j_distance = delta_j * d
        distances.append((step_height, i_distance, j_distance))

    # Only compute shadows for the actual chunk area in a super_array
    # We don't care about the overlap areas in the output array, they just get
    # overwritten by the nodata value
    if overlap > 0:
        y_start = overlap - 1
        y_end = rows - overlap
        x_start = overlap - 1
        x_end = cols - overlap
    else:
        y_start = 0
        y_end = rows
        x_start = 0
        x_end = cols

    for i in range(y_start, y_end):
        for j in range(x_start, x_end):

            point_elev = in_array[i, j]  # the point we want to determine if in shadow

            for step in range(1, max_steps):  # start at a step of 1- a point cannot be shadowed by itself

                # No need to continue if it's already shadowed
                if shadow_array[i, j] == 0:
                    already_shadowed += 1
                    # print("shadow break")
                    break

                critical_height = distances[step-1][0] + point_elev

                # idx_i/j are indices of array corresponding to current position + y/x distances
                idx_i = int(round(i + distances[step-1][1]))
                idx_j = int(round(j + distances[step-1][2]))

                in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
                in_height = critical_height < max_elev

                if in_bounds and in_height:
                    next_elev = in_array[idx_i, idx_j]
                    # Bail out if we hit a nodata area
                    if next_elev == nodata or next_elev == np.nan:
                        break

                    if next_elev > point_elev and next_elev > critical_height:
                        shadow_array[i, j] = 0

                        # set all array indices in between our found shadowing index and the source index to shadowed
                        for step2 in range(1, step):
                            i2 = int(round(i + distances[step2-1][1]))
                            j2 = int(round(j + distances[step2-1][2]))
                            shadow_array[i2, j2] = 0

                        break  # We're done with this point, move on to the next

    return shadow_array


def TPI(in_array, s_nodata, radius):
    '''
    Returns an array of the Topographic Position Index of each cell (the
    difference between the cell and the average of its neighbors). AKA, a
    high-pass mean filter.
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    radius:         The radius, in cells, of the neighborhood used for the
                    average (uses a circular window of diameter 2 * radius + 1
                    to account for the subject pixel)
    '''

    # Annulus (donut) kernel, for future advanced TPI calculations
    # i_radius = radius/2
    # o_mask = x**2 + y**2 > radius**2
    # i_mask = x**2 + y**2 < i_radius**2
    # mask = np.logical_or(o_mask, i_mask)
    # valid_entries = mask.size - np.count_nonzero(mask)
    # kernel = np.ones((diameter, diameter)) / (valid_entries)
    # kernel[mask] = 0

    # Use the blur_mean method to calculate average of neighbors
    circular_mean = blur_mean(in_array, s_nodata, radius)
    return in_array - circular_mean


def mdenoise(in_array, s_nodata, cell_size, t, n, v, tile=None, verbose=False):
    '''
    Smoothes an array of elevations using the mesh denoise algorithm by Sun et
    al (2007), Fast and Effective Feature-Preserving Mesh Denoising
    (http://www.cs.cf.ac.uk/meshfiltering/index_files/Page342.htm).
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    t:              Threshold parameter for mdenoise.exe; range [0,1]
    n:              Normal updating iterations for mdenoise; try between 10
                    and 50. Larger values increase smoothing effect and runtime
    v:              Vertext updating iterations for mdenoise; try between 10
                    and 90. Appears to affect what level of detail is smoothed
                    away.
    tile:           The name of the tile (optional). Used to differentiate the
                    temporary files' filenames.
    '''
    # Implements mdenoise algorithm by Sun et al (2007)
    # The stock mdenoise.exe runs out of memory with a window size of somewhere
    # between 1500 and 2000 (with a filter size of 15, which gives a total
    # array of window + 4 * filter). Recompiling mdenoise from source on a
    # 64-bit platform may solve this.

    # Really should just bite the bullet and rewrite/link mdenoise into
    # python so that we can just pass the np.array directly. May run into some
    # licensing restrictions by linking, as mdenoise is GPL.

    # Nodata Masking:
    # nd values get passed to mdenoise via array
    # Return array has nd values mostly intact except for some weird burrs that
    # need to be trimmed for sake of contours (done in ProcessSuperArray() by
    # copying over nodata values as mask, not in here)

    # Should be multiprocessing safe; source and target files identified with
    # pid or tile in the file name, no need for locking.

    # If the file is empty (all NoData), just return the original array
    if in_array.mean() == s_nodata:
        return in_array

    # Set up paths
    temp_dir = tempfile.gettempdir()
    if tile:  # If we have a tile name, use that for differentiator
        temp_s_path = os.path.join(temp_dir, "mesh_source_{}.asc".format(tile))
        temp_t_path = os.path.join(temp_dir, "mesh_target_{}.asc".format(tile))
    else:  # Otherwise, use the pid
        pid = mp.current_process().pid
        temp_s_path = os.path.join(temp_dir, "mesh_source_{}.asc".format(pid))
        temp_t_path = os.path.join(temp_dir, "mesh_target_{}.asc".format(pid))

    # Write array to temporary ESRI ascii file
    WriteASC(in_array, temp_s_path, 1, 1, cell_size, s_nodata)

    # Call mdenoise on temporary file
    args = (settings.MDENOISE_PATH, "-i", temp_s_path, "-t", str(t), "-n", str(n),
            "-v", str(v), "-o", temp_t_path)
    mdenoise_output = subprocess.check_output(args, shell=False,
                                              universal_newlines=True)
    if verbose:
        print(mdenoise_output)

    # Read resulting asc file into numpy array, pass back to caller
    temp_t_fh = gdal.Open(temp_t_path, gdal.GA_ReadOnly)
    temp_t_band = temp_t_fh.GetRasterBand(1)
    mdenoised_array = temp_t_band.ReadAsArray()

    # Clean up after ourselves
    temp_t_fh = None
    temp_t_band = None

    with contextlib.suppress(FileNotFoundError):
        os.remove(temp_s_path)
        os.remove(temp_t_path)

    return mdenoised_array
