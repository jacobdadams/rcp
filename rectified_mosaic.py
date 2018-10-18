#*****************************************************************************
#
#  Project:  Automatic Mosaicing of Rectified, Collared Historic Aerial Imagery
#  Purpose:  Automatically tile and merge a directory of overlapping
#            georectified aearial images, choosing from overlapping tiles based
#            on distance to the center of the parent image and the amount of
#            NoData points in the tile in order to remove collars, edges, and
#            areas of most probable distortion.
#  Author:   Jacob Adams, jacob.adams@cachecounty.org
#
#*****************************************************************************
# MIT License
#
# Copyright (c) 2018 Cache County
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#*****************************************************************************

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import os
import csv
import numpy as np


def ceildiv(a, b):
    '''
    Ceiling division, from user dlitz, https://stackoverflow.com/a/17511341/674039
    '''
    return -(-a // b)


def GetBoundingBox(in_path):
    s_fh = gdal.Open(in_path, gdal.GA_ReadOnly)
    trans = s_fh.GetGeoTransform()
    ulx = trans[0]
    uly = trans[3]
    # Calculate lower right x/y with rows/cols * cell size + origin
    lrx = s_fh.RasterXSize * trans[1] + ulx
    lry = s_fh.RasterYSize * trans[5] + uly

    s_fh = None

    return (ulx, uly, lrx, lry)


def CreateFishnetIndices(ulx, uly, lrx, lry, dimension, pixels=False, pixel_size=2.5):
    '''
    Creates a list of indicies that cover the given bounding box (may extend
    beyond the lrx/y point) with a spacing specified by 'dimension'.
    If pixels is true, assumes dimensions are in pixels and uses pixel_size.
    Otherwise, dimension is in raster coordinate system.
    '''

    chunks = []

    ref_width = lrx - ulx
    ref_height = uly - lry
    if pixels:
        chunk_ref_size = dimension * pixel_size
    else:
        chunk_ref_size = dimension
    num_x_chunks = int(ceildiv(ref_width, chunk_ref_size))
    num_y_chunks = int(ceildiv(ref_height, chunk_ref_size))
    for y_chunk in range(0, num_y_chunks):
        for x_chunk in range(0, num_x_chunks):
            x_index = x_chunk
            y_index = y_chunk
            chunk_ulx = ulx + (chunk_ref_size * x_index)
            chunk_uly = uly + (-chunk_ref_size * y_index)
            chunk_lrx = ulx + (chunk_ref_size * (x_index + 1))
            chunk_lry = uly + (-chunk_ref_size * (y_index + 1))
            chunks.append((x_index, y_index, chunk_ulx, chunk_uly, chunk_lrx,
                           chunk_lry))

    return chunks


def create_polygon(coords):
    '''
    Creates a WKT polygon from a list of coordinates
    '''
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()


def CopyTilesFromRaster(root, rastername, fishnet, shp_layer, target_dir):
    '''
    Given a fishnet of a certain size, copy any chunks of the source raster
    into individual files corresponding to the fishnet cells. Calculates
    the distance from the cell center to the raster's center, and stores in
    a shapefile containing the bounding box of each cell.

    Returns a dictionary containing the distance to center for each sub-chunk
    in the form {cell_name: (rastername, distance, nodata found in chunk) ...}
    '''

    distances = {}

    raster_path = os.path.join(root, rastername)

    # Get data from source raster
    s_fh = gdal.Open(raster_path, gdal.GA_ReadOnly)
    trans = s_fh.GetGeoTransform()
    projection = s_fh.GetProjection()
    band1 = s_fh.GetRasterBand(1)
    s_nodata = band1.GetNoDataValue()
    bands = s_fh.RasterCount
    raster_xmin = trans[0]
    raster_ymax = trans[3]
    raster_xwidth = trans[1]
    raster_yheight = trans[5]
    driver = s_fh.GetDriver()
    lzw_opts = ["compress=lzw", "tiled=yes"]
    band1 = None

    # Calculate lower right x/y with rows/cols * cell size + origin
    raster_xmax = s_fh.RasterXSize * raster_xwidth + raster_xmin
    raster_ymin = s_fh.RasterYSize * raster_yheight + raster_ymax

    # Calculate raster middle
    raster_xmid = (s_fh.RasterXSize / 2.) * raster_xwidth + raster_xmin
    raster_ymid = (s_fh.RasterYSize / 2.) * raster_yheight + raster_ymax

    # Loop through the cells in the fishnet, copying over any relevant bits of
    # raster to new subchunks.
    for cell in fishnet:

        cell_name = "{}-{}".format(cell[0], cell[1])
        cell_xmin = cell[2]
        cell_xmax = cell[4]
        cell_ymin = cell[5]
        cell_ymax = cell[3]

        cell_xmid = (cell_xmax - cell_xmin) / 2. + cell_xmin
        cell_ymid = (cell_ymax - cell_ymin) / 2. + cell_ymin

        # Check to see if some part of raster is inside a given fishnet
        # cell.
        # If cell x min or max and y min or max are inside the raster
        xmin_inside = cell_xmin > raster_xmin and cell_xmin < raster_xmax
        xmax_inside = cell_xmax > raster_xmin and cell_xmax < raster_xmax
        ymin_inside = cell_ymin > raster_ymin and cell_ymin < raster_ymax
        ymax_inside = cell_ymax > raster_ymin and cell_ymax < raster_ymax
        if (xmin_inside or xmax_inside) and (ymin_inside or ymax_inside):

            # Translate cell coords to raster pixels, create a numpy array intialized to nodatas, readasarray, save as cell_raster.tif

            #print("{} {} {} {}".format(cell_xmin, raster_xmin, cell_ymax, raster_ymax))

            # Fishnet cell origin and size as pixel indices
            x_off = int((cell_xmin - raster_xmin) / raster_xwidth)
            y_off = int((cell_ymax - raster_ymax) / raster_yheight)
            # Add 5 pixels to x/y_size to handle gaps
            x_size = int((cell_xmax - cell_xmin) / raster_xwidth) + 5
            y_size = int((cell_ymin - cell_ymax) / raster_yheight) + 5

            #print("{} {} {} {}".format(x_off, y_off, x_size, y_size))

            # Values for ReadAsArray, these aren't changed later unelss
            # the border case checks change them
            # These are all in pixels
            # We are adding two to read_x/y_size to slightly overread to
            # catch small one or two pixel gaps in the combined rasters.
            read_x_off = x_off
            read_y_off = y_off
            read_x_size = x_size
            read_y_size = y_size

            # Slice values for copying read data into slice_array
            # These are the indices in the slice array where the actual
            # read data should be copied to.
            # These should be 0 and max size (ie, same dimension as
            # read_array) unelss the border case checks change them.
            sa_x_start = 0
            sa_x_end = x_size
            sa_y_start = 0
            sa_y_end = y_size

            # Edge logic
            # If read exceeds bounds of image:
            #   Adjust x/y offset to appropriate place
            #   Change slice indices
            # Checks both x and y, setting read and slice values for each dimension if
            # needed
            if x_off < 0:
                read_x_off = 0
                read_x_size = x_size + x_off  # x_off would be negative
                sa_x_start = -x_off  # shift inwards -x_off spaces
            if x_off + x_size > s_fh.RasterXSize:
                read_x_size = s_fh.RasterXSize - x_off
                sa_x_end = read_x_size  # end after read_x_size spaces

            if y_off < 0:
                read_y_off = 0
                read_y_size = y_size + y_off
                sa_y_start = -y_off
            if y_off + y_size > s_fh.RasterYSize:
                read_y_size = s_fh.RasterYSize - y_off
                sa_y_end = read_y_size

            # Set up output raster
            t_rastername = "{}_{}.tif".format(cell_name, rastername[:-4])
            #print(t_rastername)
            t_path = os.path.join(target_dir, t_rastername)
            t_fh = driver.Create(t_path, x_size, y_size, bands, gdal.GDT_Int16, options=lzw_opts)
            t_fh.SetProjection(projection)

            # TO FIX WEIRD OFFSETS:
            # Make sure tranform is set based on the top left corner of top left pixel of the source raster, not the fishnet. Using fishnet translates the whole raster to the fishnet's grid, which isn't consistent with the rasters' pixel grids
            # i.e., cell_x/ymin is not the top left corner of top left pixel of the raster

            # Translate from x/y_off (pixels) to raster's GCS
            raster_chunk_xmin = x_off * raster_xwidth + raster_xmin
            raster_chunk_ymax = y_off * raster_yheight + raster_ymax

            # Transform:
            # 0: x coord, top left corner of top left pixel
            # 1: pixel width
            # 2: 0 (for north up)
            # 3: y coord, top left corner of top left pixel
            # 4: 0 (for north up)
            # 5: pixel height
            # t_trans = (cell_xmin, raster_xwidth, 0, cell_ymax, 0, raster_yheight)
            t_trans = (raster_chunk_xmin, raster_xwidth, 0, raster_chunk_ymax, 0, raster_yheight)
            t_fh.SetGeoTransform(t_trans)

            num_nodata = 0
            # Loop through all the bands of the raster and copy to a new chunk
            for band in range(1, bands + 1):
                # Prep target band
                t_band = t_fh.GetRasterBand(band)
                t_band.SetNoDataValue(s_nodata)

                # Initialize slice array to nodata (for areas of the new chunk
                # that are outside the source raster)
                slice_array = np.full((y_size, x_size), s_nodata)

                # Read the source raster
                s_band = s_fh.GetRasterBand(band)
                read_array = s_band.ReadAsArray(read_x_off, read_y_off,
                                                read_x_size, read_y_size)

                num_nodata += (read_array == s_nodata).sum()
                # Put source raster data into appropriate place of slice array
                slice_array[sa_y_start:sa_y_end, sa_x_start:sa_x_end] = read_array

                # Write source array to disk
                t_band.WriteArray(slice_array)
                t_band = None
                s_band = None

            # Close target file handle
            t_fh = None

            # Calculate distance from cell center to raster center
            cell_center = np.array((cell_xmid, cell_ymid))
            raster_center = np.array((raster_xmid, raster_ymid))
            distance = np.linalg.norm(cell_center - raster_center)

            new_num_nodata = num_nodata / 3.

            print("{}, {}, {}, {}".format(cell_name, rastername, distance, new_num_nodata))

            # Create cell bounding boxes as shapefile, with distance from the
            # middle of the cell to the middle of it's parent raster saved as a
            # field for future evaluation
            coords = [(cell_xmin, cell_ymax),
                      (cell_xmax, cell_ymax),
                      (cell_xmax, cell_ymin),
                      (cell_xmin, cell_ymin),
                      (cell_xmin, cell_ymax)]
            defn = shp_layer.GetLayerDefn()
            feature = ogr.Feature(defn)
            feature.SetField('raster', rastername)
            feature.SetField('cell', cell_name)
            feature.SetField('d_to_cent', distance)
            feature.SetField('nodatas', new_num_nodata)
            poly = create_polygon(coords)
            geom = ogr.CreateGeometryFromWkt(poly)
            feature.SetGeometry(geom)
            shp_layer.CreateFeature(feature)
            feature = None
            poly = None
            geom = None

            distances[cell_name] = (rastername, distance, new_num_nodata)

    # close source raster
    s_fh = None

    return distances


def TileRectifiedRasters(rectified_dir, shp_path, tiled_dir, fishnet_size):
    '''
    Tiles all the rasters in rectified_dir into tiles based on a fishnet
    starting at the upper left of all the rasters and that has cells of
    fishnet_size, saving them in tiled_dir. Each fishnet cell will have
    multiple tiles associated with it if two or more rasters overlap. The
    following information is calculated for each tile, stored in the fishnet
    shapefile, and returned from the method: the parent raster, the fishnet
    cell index, the distance from the center of the tile to the center of the
    parent raster, and the number of nodata pixels in the tile.

    Returns: A list of dictionaries containing the tile information like so:
    [{cell_index: (rastername, distance, nodatas)}, {}, ...]
    '''

    #directory = r'e:\a_imagery\1981\rectified'

    # Loop through rectified rasters, check for ul/lr x/y to get bounding box
    # ulx is the smallest x value, so we set it high and check if the current
    # one is lower
    ulx = 999999999
    # uly is the largest y, so we set low and check if greater
    uly = 0
    # lrx is largest x, so we set low and check if greater
    lrx = 0
    # lry is smallest y, so we set high and check if lower
    lry = 999999999
    for root, dirs, files in os.walk(rectified_dir):
        for fname in files:
            if fname[-4:] == ".tif":
                img_path = os.path.join(root, fname)
                bounds = GetBoundingBox(img_path)
                if bounds[0] < ulx:
                    ulx = bounds[0]
                if bounds[1] > uly:
                    uly = bounds[1]
                if bounds[2] > lrx:
                    lrx = bounds[2]
                if bounds[3] < lry:
                    lry = bounds[3]
    # print("{}, {}; {}, {}".format(ulx, uly, lrx, lry))

    # Create tiling scheme
    fishnet = CreateFishnetIndices(ulx, uly, lrx, lry, fishnet_size)
    # for cell in fishnet:
    #     print(cell)

    # Set up fishnet polygons shapefile
    #poly_shp = r'e:\a_imagery\1981\00fishnet.shp'
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = shp_driver.CreateDataSource(shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(102742)
    layer = shp_ds.CreateLayer('', srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn('raster', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('cell', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('d_to_cent', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('nodatas', ogr.OFTReal))

    # Retiled directory
    #tile_dir = r'e:\a_imagery\1981\tiled'

    # list containing all chunk dictionaries
    all_cells = []

    # Loop through rectified rasters, create tiles named by index
    for root, dirs, files in os.walk(rectified_dir):
        for fname in files:
            if fname[-4:] == ".tif":
                #print(fname)

                distances = CopyTilesFromRaster(root, fname, fishnet, layer,
                                                tiled_dir)
                all_cells.append(distances)

                # # Update add or overwrite cell in chunks dictionary if it isn't
                # # presnt already or if the distance is shorter than the current one
                # # and the new chunk has fewer nodata values
                # for cell, rname_distance in distances.items():
                #     if cell in chunks:  # Is there a chunk for this cell already?
                #         if rname_distance[1] < chunks[cell][1]:  # is this one closer to the center of the raster than the existing one?
                #             if rname_distance[2] <= chunks[cell][2]:  # does this one have fewer nodatas (or the same as) that the existing one?
                #                 chunks[cell] = rname_distance
                #     elif cell not in chunks:
                #         chunks[cell] = rname_distance

    # Cleanup shapefile handles
    layer = None
    shp_ds = None

    return all_cells


def ReadChunkFromShapefile(shp_path):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shape_s_dh = driver.Open(shp_path, 0)
    layer = shape_s_dh.GetLayer()

    #[{cell_index: (rastername, distance, nodatas)}, {}, ...]
    cells = []
    for feature in layer:
        cell_index = feature.GetField("cell")
        rastername = feature.GetField("raster")
        distance = feature.GetField("d_to_cent")
        nodatas = feature.GetField("nodatas")
        celldict = {}
        celldict[cell_index] = (rastername, distance, nodatas)
        cells.append(celldict)
    layer = None
    shape_s_dh = None

    return cells


if "__main__" in __name__:
    directory = r'e:\a_imagery\1981\1_rectified'
    poly_shp = r'e:\a_imagery\1981\00fishnet.shp'
    tile_dir = r'e:\a_imagery\1981\tiled'
    csv_path = r'e:\a_imagery\1981\00cells.csv'
    # directory = r'f:\1978plats'
    # poly_shp = r'f:\1978plats\00fishnet.shp'
    # tile_dir = r'f:\1978plats\tiled'
    fishnet_size = 750
    tile = True

    # Retile if needed; otherwise, just read the shapefile
    if tile:
        all_cells = TileRectifiedRasters(directory, poly_shp, tile_dir, fishnet_size)
    else:
        all_cells = ReadChunkFromShapefile(poly_shp)

    # dictionary containing rasternames, chunknames, distances, nodata counts
    chunks = {}
    # Update add or overwrite cell in chunks dictionary if it isn't
    # presnt already or if the distance is shorter than the current one
    # and the new chunk has fewer nodata values
    for cell in all_cells:
        for cell_number, cell_info in cell.items():
            if cell_number in chunks:  # Is there a chunk for this cell already?
                if cell_info[1] < chunks[cell_number][1]:  # is this one closer to the center of the raster than the existing one?
                    if cell_info[2] <= chunks[cell_number][2]:  # does this one have fewer nodatas (or the same as) that the existing one?
                        chunks[cell_number] = cell_info
            elif cell_number not in chunks:
                chunks[cell_number] = cell_info

    # Do a second pass, so that when we build a vrt and the most desirable
    # chunk has an area of nodata but there's a less-desirable chunk that
    # covers part of that nodata, gdalbuildvrt will add this second chunk
    # underneath it and honor the nodata setting of the upper, more desirable
    # chunk.
    second_chunks = {}
    for cell in all_cells:
        for cell_number, cell_info in cell.items():
            if cell_number not in chunks:  # can't be in first pass dictionary
                if cell_number in second_chunks:
                    if cell_info[1] < second_chunks[cell_number][1]:  # is this one closer to the center of the raster than the existing one?
                        if cell_info[2] <= second_chunks[cell_number][2]:  # does this one have fewer nodatas (or the same as) that the existing one?
                            second_chunks[cell_number] = cell_info
                else:
                    second_chunks[cell_number] = cell_info

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write second-pass chunks first, so that first pass will be seen later
        # and be placed on top by gdalbuidvrt
        for key, value in second_chunks.items():
            chunk_name = "{}_{}".format(key, value[0])
            chunk_path = os.path.join(tile_dir, chunk_name)
            writer.writerow([chunk_path])
        # Now first-pass, most desirable chunks:
        for key, value in chunks.items():
            chunk_name = "{}_{}".format(key, value[0])
            chunk_path = os.path.join(tile_dir, chunk_name)
            writer.writerow([chunk_path])
