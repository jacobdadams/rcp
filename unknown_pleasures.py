import math
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

in_dem_path = "c:\\temp\\gis\\dem_state.tif"

gdal.UseExceptions()

# Get source file metadata (dimensions, driver, proj, cell size, nodata)
s_fh = gdal.Open(in_dem_path, gdal.GA_ReadOnly)
rows = s_fh.RasterYSize
cols = s_fh.RasterXSize
driver = s_fh.GetDriver()
s_band = s_fh.GetRasterBand(1)
nodata = s_band.GetNoDataValue()
# Get source georeference info
transform = s_fh.GetGeoTransform()
projection = s_fh.GetProjection()
#cell_size = abs(transform[5])  # Assumes square pixels where height=width
#s_nodata = s_band.GetNoDataValue()

source_x_origin = transform[0]
source_y_origin = transform[3]
pixel_width = transform[1]
pixel_height = -transform[5]
#print("x orig: {} y orig: {} width: {} height: {}".format(source_x_origin, source_y_origin, pixel_width, pixel_height))

data = s_band.ReadAsArray(0, 0, cols, rows)
masked_data = np.ma.masked_equal(data, nodata)

data_min = masked_data.min()

# x vals = cols = j, y vals = rows = i
# coords are in x, y (easting, northing)
# loop through rows- every 100 feet
#   loop through each column- every 100 feet
# orig drawing has 80 lines
# UL: 1,467,208, 3,837,848
# LR: 1,594,422, 3,756,858
# x between 1,594,000 and 1,467,000 => 127 @ 1000'
# y between 3,837,000 and 3,756,000 => 81 @ 1000'

# Original
# read_x_right =   1594000
# read_x_left =    1467000
# read_y_top =     3837000
# read_y_bottom =  3756000

# Extend all
# read_x_right =   1625000
# read_x_left =    1467000
# read_y_top =     3870000
# read_y_bottom =  3710000

# # Extend X
# read_x_right =   1625000
# read_x_left =    1467000
# read_y_top =     3837000
# read_y_bottom =  3756000
#
# # horizontal gap
# x_gap = 500
# # vertical gap
# y_gap = 1000

## original unrotated method
# # A list of lists of elevations x_gap apart horizontally. Each sub-list is y_gap apart vertically
# row_elevs_list = []
#
# # coordinates: easting = x = cols, northing = y = rows
#
# # Loop through rows (y), then each column (x) in each row (in supplied coords)
# # add x/y_gap to range to add the last value to the range
# for y in range(read_y_bottom, read_y_top + y_gap, y_gap):
#     row_elevs = []  # each entry should be elevation at that one point
#     for x in range(read_x_left, read_x_right + x_gap, x_gap):
#
#         # Get the raster indexes of the supplied coords
#         # x is x coordinate, y is y coordinate in supplied coord system
#         source_x_index = int((x - source_x_origin) / pixel_width)
#         source_y_index = int((source_y_origin - y) / pixel_height)
#
#         # Read from raster, which is accessed via [row, col]
#         elev = data[source_y_index, source_x_index]
#         row_elevs.append(elev)
#     row_elevs_list.append(row_elevs)

# set by number of rows/cols desired instead of specifying absolute extent
# num_rows = int(read_y_top - read_y_bottom) / y_gap
# num_cols = int(read_x_right - read_x_left) / x_gap
#
# rotate = 5.0
# DEG_TO_RAD = math.pi / 180.0
# rotate_rad = rotate * DEG_TO_RAD

# TEMP FOR TESTING, CAN DELETE
# num_rows = 10
# num_cols = 10



# Lower y_gap -> higher 'viewpoint'
# print_offset acts as a scaling factor. The higher the offset, the less pronounced the terrain
#   200 seems a little excessive, 2000 is very flat

# Cache Front
# Origin x: 1,535,295
# Origin y: 3,870,610
# Rotate: 90

# cache_front
# origin = (1535295, 3870610)
# rotate = 90
# width = 140000
# height = 50000

# # Wellsvilles
# origin = (1538200, 3738970)
# rotate = 270
# width = 80000
# height = 80000

# valley south
origin = (1619695, 3840010)
rotate = 180
width = 160000
height = 140000

# horizontal gap
x_gap = 250
# vertical gap
y_gap = 1000

num_rows = int(height / y_gap)
num_cols = int(width / x_gap)

print("Rows: {}    Cols: {}".format(num_rows, num_cols))

print_offset = 300

# Y values for each row
row_y_indexes = []
# starting y value for nth row: previous starting y value + cos(theta)*y_gap
#   next y value in nth row: previous y value - cos(90-theta)*x_gap
row_y_origin = origin[1]
for row in range(0, num_rows):  # build list of y-values in coord system for each row
    row_ys = []  # list of y vals for this row
    row_ys.append(int(row_y_origin))  # first y-val is the row origin point

    # calculate the next y values for each column in this row
    prev_y_val = row_y_origin
    for col in range(1, num_cols):  # start at 1 because we already added the origin
        y_val = prev_y_val - math.cos(math.radians(90 - rotate)) * x_gap
        row_ys.append(int(y_val))  # add it to the list
        prev_y_val = y_val  # set the y val for the next col in this row

    # Add the list of y values for this row to the list of rows
    row_y_indexes.append(row_ys)

    # Set the y value for the next row
    row_y_origin = row_y_origin + math.cos(math.radians(rotate)) * y_gap

# X values for each row
row_x_indexes = []
# starting x value for nth row: previous starting x value + sin(theta)*y_gap
#   next y value in nth row: previous x value + sin(90-theta)*x_gap
row_x_origin = origin[0]
for row in range(0, num_rows):  # build list of x-values in coord system for each row
    row_xs = []  # list of x vals for this row
    row_xs.append(int(row_x_origin))  # first x-val is the row origin point

    # calculate the next x values for each column in this row
    prev_x_val = row_x_origin
    for col in range(1, num_cols):  # start at 1 because we already added the origin
        x_val = prev_x_val + math.sin(math.radians(90 - rotate)) * x_gap
        row_xs.append(int(x_val))  # add it to the list
        prev_x_val = x_val  # set the x val for the next col in this row

    # Add the list of x values for this row to the list of rows
    row_x_indexes.append(row_xs)

    # Set the x value for the next row
    row_x_origin = row_x_origin + math.sin(math.radians(rotate)) * y_gap

# for row in row_y_indexes:
#     print(row)
# for row in row_x_indexes:
#     print(row)

# Merge y, x values into tuples for each point for each row
coord_rows = []
for row_y, row_x in zip(row_y_indexes, row_x_indexes):
    current_row = []
    for y, x in zip(row_y, row_x):
        current_row.append((y, x))
    coord_rows.append(current_row)
# for row in coord_rows:
#     print(row)

# A list of lists of elevations x_gap apart horizontally. Each sub-list is y_gap apart vertically. The dataset has already been rotated
row_elevs_list = []

# coordinates: easting = x = cols, northing = y = rows

# Loop through list of coordinate rows (each item is a (y,x) tuple)
for row in coord_rows:
    row_elevs = []
    for coord_pair in row:
        x = coord_pair[1]
        y = coord_pair[0]
        # Get the raster indexes of the supplied coords
        # x is x coordinate, y is y coordinate in supplied coord system
        source_x_index = int((x - source_x_origin) / pixel_width)
        source_y_index = int((source_y_origin - y) / pixel_height)

        # Read from raster, which is accessed via [row, col]
        try:
            elev = masked_data[source_y_index, source_x_index]
            if elev < data_min:
                row_elevs.append(data_min)
            else:
                row_elevs.append(elev)
        except IndexError:
            row_elevs.append(data_min)
    row_elevs_list.append(row_elevs)


elevs = [r for r in row_elevs_list[0]]

# shift each row of elevations up by i * print_offset for printing
new_row_elevs_list = []
for i in range(0, num_rows):
    offset = i * print_offset
    new_row_elevs = []
    for val in row_elevs_list[i]:
        new_row_elevs.append(val + offset)
    new_row_elevs_list.append(new_row_elevs)

# Print it out
y_start = row_elevs_list[0][0]
x_vals = range(0, len(new_row_elevs_list[0]))
for stripe in new_row_elevs_list[::-1]:
    # Working from the back, add the polygon and the line for each slice
    plt.fill_between(x_vals, stripe, facecolor = 'black', edgecolor = 'white')
    #plt.plot(stripe, color='white')
plt.ylim(ymin=data_min)
plt.axis('off')
plt.show()

#print(row_list[0])
#print(row_elevs_list[0])

# for reading given a list of points in the same coords as the source_file:
# for point in points_list:
#     col = int((point[0] - source_x_origin) / pixel_width)
#     row = int((source_y_origin - point[1]) / pixel_height)
#
#     print("{}, {}, {}".format(row, col, data[row][col]))

# Close source file handle
s_band = None
s_fh = None
