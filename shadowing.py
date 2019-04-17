from numba.pycc import CC
import numpy as np
import math

cc = CC('shadowing')


@cc.export('shadows', "u1[:,:](f4[:,:],f8,f8,f8)")
def shadows(in_array, az, alt, res):
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
    max_steps = 150

    for i in range(0, rows):
        for j in range(0, cols):

            point_elev = in_array[i, j]  # the point we want to determine if in shadow
            # start calculating next point from the source point

            for p in range(0, max_steps):
                # Figure out next point along the path
                next_i = i + delta_i * p * mult_size
                next_j = j + delta_j * p * mult_size

                # We need integar indexes for the array
                idx_i = int(round(next_i))
                idx_j = int(round(next_j))

                # No need to continue if it's already shadowed
                if shadow_array[i, j] == 0:
                    break

                # distance for elevation check is distance in cells (idx_i/j), not distance along the path
                # critical height is the elevation that is directly in the path of the sun at given alt/az
                idx_distance = math.sqrt((i - idx_i)**2 + (j - idx_j)**2)
                critical_height = idx_distance * tanaltrad * res + point_elev

                in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
                in_height = critical_height < max_elev

                if in_bounds and in_height:
                    next_elev = in_array[idx_i, idx_j]
                    if next_elev > point_elev and next_elev > critical_height:
                        shadow_array[i, j] = 0
                        break  # We're done with this point, move on to the next

    return shadow_array


if __name__ == '__main__':
    cc.compile()
