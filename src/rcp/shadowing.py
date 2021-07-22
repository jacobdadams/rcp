from numba.pycc import CC
import numpy as np
import math

cc = CC('shadowing')
#cc.verbose = True
#print(cc.target_cpu)
cc.target_cpu = "host"


@cc.export('shadows', "u1[:,:](f8[:,:],f8,f8,f8,u4,f8)")
# @cc.export('shadows', "u1[:,:](f4[:,:],f8,f8,f8,u4,f8)")
def shadows(in_array, az, alt, res, overlap, nodata):
    # Rows = i = y values, cols = j = x values
    rows = in_array.shape[0]
    cols = in_array.shape[1]
    shadow_array = np.ones(in_array.shape, dtype=np.uint8)  # init to 1 (not shadowed), change to 0 if shadowed
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

            # # Bail out if point is nodata
            # if point_elev == nodata:
            #     break

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
                    if next_elev == nodata:
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


if __name__ == '__main__':
    cc.compile()
