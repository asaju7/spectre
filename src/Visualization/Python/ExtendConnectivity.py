#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import h5py
import logging
import numpy as np
import sys


def generate_grid_point_dictionary(coord_data_x, coord_data_y, coord_data_z):

    gp_dictionary = {}
    for i in range(len(coord_data_x)):
        gp_dictionary.update({
            (coord_data_x[i], coord_data_y[i], coord_data_z[i]):
            i
        })
    return gp_dictionary


def sort_and_order(data_list):

    ordered_coords = []
    ordered_coords.append(data_list[0])
    for i in range(1, len(data_list)):
        k = 0
        for j in ordered_coords:
            k += 1
            if (j == data_list[i]):
                break
            elif (k == len(ordered_coords)):
                ordered_coords.append(data_list[i])
    ordered_coords.sort()
    return ordered_coords


def build_connectivity_by_element(sorted_x, sorted_y, sorted_z):

    connectivity_as_tuples = []
    for k in range(len(sorted_z) - 1):
        for j in range(len(sorted_y) - 1):
            for i in range(len(sorted_x) - 1):
                connectivity_as_tuples.append(
                    (sorted_x[i], sorted_y[j], sorted_z[k]))
                connectivity_as_tuples.append(
                    (sorted_x[i + 1], sorted_y[j], sorted_z[k]))
                connectivity_as_tuples.append(
                    (sorted_x[i + 1], sorted_y[j + 1], sorted_z[k]))
                connectivity_as_tuples.append(
                    (sorted_x[i], sorted_y[j + 1], sorted_z[k]))
                connectivity_as_tuples.append(
                    (sorted_x[i], sorted_y[j], sorted_z[k + 1]))
                connectivity_as_tuples.append(
                    (sorted_x[i + 1], sorted_y[j], sorted_z[k + 1]))
                connectivity_as_tuples.append(
                    (sorted_x[i + 1], sorted_y[j + 1], sorted_z[k + 1]))
                connectivity_as_tuples.append(
                    (sorted_x[i], sorted_y[j + 1], sorted_z[k + 1]))
    return connectivity_as_tuples


def new_connectivity(inertial_coords_x, inertial_coords_y, inertial_coords_z):

    gp_dictionary = generate_grid_point_dictionary(inertial_coords_x,
                                                   inertial_coords_y,
                                                   inertial_coords_z)
    ordered_x = sort_and_order(inertial_coords_x)
    ordered_y = sort_and_order(inertial_coords_y)
    ordered_z = sort_and_order(inertial_coords_z)
    connectivity_of_tuples = build_connectivity_by_element(
        ordered_x, ordered_y, ordered_z)
    connectivity = []
    for i in connectivity_of_tuples:
        connectivity.append(gp_dictionary[i])
    return connectivity


def fix_connectivity(input_file, subfile_name):

    if sys.version_info < (3, 0):
        logging.warning("You are attempting to run this script with "
                        "python 2, which is deprecated. GenerateXdmf.py might "
                        "hang or run very slowly using python 2. Please use "
                        "python 3 instead.")

    path_with_file_name = input_file
    with h5py.File(path_with_file_name, 'r+') as ds:
        data = ds[subfile_name + ".vol"]
        observation_keys = list(data.keys())
        for i in range(len(observation_keys)):
            data_y = np.array(ds[subfile_name + ".vol/" + observation_keys[i] +
                                 "/InertialCoordinates_y"])
            data_x = np.array(ds[subfile_name + ".vol/" + observation_keys[i] +
                                 "/InertialCoordinates_x"])
            data_z = np.array(ds[subfile_name + ".vol/" + observation_keys[i] +
                                 "/InertialCoordinates_z"])
            new_connect = new_connectivity(data_x, data_y, data_z)
            del ds[subfile_name + ".vol/" + observation_keys[i] +
                   "/connectivity"]
            ds.create_dataset(subfile_name + ".vol/" + observation_keys[i] +
                              "/connectivity",
                              data=new_connect)
        ds.close()


def parse_args():
    """
    Parse the command line arguments
    """
    import argparse as ap
    parser = ap.ArgumentParser(
        description=
        "Connects the blocks of Gauss by updating the connectivity of a single"
        " .h5 file. ",
        formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-file',
        required=True,
        help=
        "The input file with the .h5 extension where the connectivity will be "
        "updated. It must be a single .h5 file. Use the CombineH5 executable "
        "to combine multiple h5 files first.")
    parser.add_argument(
        '--subfile-name',
        required=True,
        help="Name of the volume data subfile in the H5 files, excluding the "
        "'.vol' extension")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_args = parse_args()
    fix_connectivity(**vars(input_args))
