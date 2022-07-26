#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import h5py
import logging
import numpy as np
import sys

import time


def generate_grid_point_dictionary(coord_data_x, coord_data_y, coord_data_z):

    print("Starting generate_grid_point_dictionary")
    start = time.time()

    grid_point_dictionary = {}
    for i in range(len(coord_data_x)):
        grid_point_dictionary.update({
            (coord_data_x[i], coord_data_y[i], coord_data_z[i]):
            i
        })

    end = time.time()
    print("generate_grid_point_dictionary runtime: " + str(end - start))
    return grid_point_dictionary


def sort_and_order(data_list):

    print("Starting sort_and_order")
    start = time.time()

    ordered_coords = []
    data_list.sort()
    ordered_coords.append(data_list[0])
    for i in range(1, len(data_list)):
        if data_list[i] == ordered_coords[-1]:
            continue
        else:
            ordered_coords.append(data_list[i])

    end = time.time()
    print("sort_and_order runtime: " + str(end - start))
    return ordered_coords


def build_connectivity_by_element(sorted_x, sorted_y, sorted_z):

    print("Starting build_connectivity_by_element")
    start = time.time()

    connectivity_as_tuples = []
    for k in range(len(sorted_z) - 1):
        for j in range(len(sorted_y) - 1):
            for i in range(len(sorted_x) - 1):
                connectivity_as_tuples.extend([
                    (sorted_x[i], sorted_y[j], sorted_z[k]),
                    (sorted_x[i + 1], sorted_y[j], sorted_z[k]),
                    (sorted_x[i + 1], sorted_y[j + 1], sorted_z[k]),
                    (sorted_x[i], sorted_y[j + 1], sorted_z[k]),
                    (sorted_x[i], sorted_y[j], sorted_z[k + 1]),
                    (sorted_x[i + 1], sorted_y[j], sorted_z[k + 1]),
                    (sorted_x[i + 1], sorted_y[j + 1], sorted_z[k + 1]),
                    (sorted_x[i], sorted_y[j + 1], sorted_z[k + 1])
                ])

    end = time.time()
    print("build_connectivity_by_element runtime: " + str(end - start))
    return connectivity_as_tuples


def generate_new_connectivity(inertial_coords_x, inertial_coords_y,
                              inertial_coords_z):

    print("Starting generate_new_connectivity")
    start = time.time()

    grid_point_dictionary = generate_grid_point_dictionary(
        inertial_coords_x, inertial_coords_y, inertial_coords_z)
    ordered_x = sort_and_order(inertial_coords_x)
    ordered_y = sort_and_order(inertial_coords_y)
    ordered_z = sort_and_order(inertial_coords_z)
    connectivity_of_tuples = build_connectivity_by_element(
        ordered_x, ordered_y, ordered_z)
    connectivity = []
    j = 0
    for i in connectivity_of_tuples:
        # print(i)
        if i == connectivity_of_tuples[0]:
            continue
        else:
            connectivity.append(grid_point_dictionary[i])

    end = time.time()
    print("generate_new_connectivity runtime: " + str(end - start))
    return connectivity


def extend_connectivity(input_file):

    if sys.version_info < (3, 0):
        logging.warning("You are attempting to run this script with "
                        "python 2, which is deprecated. GenerateXdmf.py might "
                        "hang or run very slowly using python 2. Please use "
                        "python 3 instead.")

    with h5py.File(input_file, 'r+') as dataset:
        subfile_keys = list(dataset.keys())
        subfile_name = subfile_keys[0]
        data = dataset[subfile_name]
        observation_keys = list(data.keys())
        for i in range(len(observation_keys)):
            print("Starting observation key number " + str(i))  ############
            data_y = np.array(
                dataset[subfile_name + "/" + observation_keys[i] +
                        "/InertialCoordinates_y"])
            data_x = np.array(
                dataset[subfile_name + "/" + observation_keys[i] +
                        "/InertialCoordinates_x"])
            data_z = np.array(
                dataset[subfile_name + "/" + observation_keys[i] +
                        "/InertialCoordinates_z"])
            new_connect = generate_new_connectivity(data_x, data_y, data_z)
            del dataset[subfile_name + "/" + observation_keys[i] +
                        "/connectivity"]
            dataset.create_dataset(subfile_name + "/" + observation_keys[i] +
                                   "/connectivity",
                                   data=new_connect)
        dataset.close()


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
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_args = parse_args()
    extend_connectivity(**vars(input_args))
