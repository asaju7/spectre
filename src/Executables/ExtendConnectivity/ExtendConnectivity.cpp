// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <boost/program_options.hpp>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"

#include <iostream>
#include <map>

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

std::pair<Mesh<3>, std::string> generate_mesh(const h5::VolumeData& volume_file,
                                              const size_t& single_id,
                                              const size_t element_number) {
  // MAKE SURE TO MAKE THE DIM GENERAL
  auto bases = volume_file.get_bases(single_id);
  std::array<Spectral::Basis, 3> basis_array = {
      Spectral::to_basis(bases[element_number][0]),
      Spectral::to_basis(bases[0][1]),
      Spectral::to_basis(bases[element_number][2])};
  // std::cout << bases << '\n' << bases.size() << '\n';

  auto quadratures = volume_file.get_quadratures(single_id);
  std::array<Spectral::Quadrature, 3> quadrature_array = {
      Spectral::to_quadrature(quadratures[element_number][0]),
      Spectral::to_quadrature(quadratures[element_number][1]),
      Spectral::to_quadrature(quadratures[element_number][2])};
  // std::cout << quadratures << '\n' << quadratures.size() << '\n';

  auto extents = volume_file.get_extents(single_id);
  std::array<size_t, 3> extents_array = {extents[element_number][0],
                                         extents[element_number][1],
                                         extents[element_number][2]};
  // std::cout << extents << '\n' << extents.size() << '\n';

  auto grid_names = volume_file.get_grid_names(single_id);
  std::string grid_names_array = grid_names[element_number];
  // std::cout << grid_names_array << '\n';
  // std::cout << grid_names << '\n';

  std::cout << "Finished generate_mesh" << '\n';
  Mesh<3> element_mesh = Mesh<3>{extents_array, basis_array, quadrature_array};
  std::pair<Mesh<3>, std::string> output(element_mesh, grid_names_array);
  return output;
}

// Compute Block-logical coordinates
tnsr::I<DataVector, 3, Frame::BlockLogical> block_logical_coords(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& new_coords,
    const std::string& element_id) {
  std::string R_x_string(1, element_id[6]);  // HARDCODE
  double R_x = std::stod(R_x_string);
  std::string R_y_string(1, element_id[11]);  // HARDCODE
  double R_y = std::stod(R_y_string);
  std::string R_z_string(1, element_id[16]);  // HARDCODE
  double R_z = std::stod(R_z_string);
  double N_x = pow(2, R_x);
  double N_y = pow(2, R_y);
  double N_z = pow(2, R_z);
  std::string index_x_string(1, element_id[8]);   // HARDCODE
  std::string index_y_string(1, element_id[13]);  // HARDCODE
  std::string index_z_string(1, element_id[18]);  // HARDCODE
  int index_x = std::stoi(index_x_string);
  int index_y = std::stoi(index_y_string);
  int index_z = std::stoi(index_z_string);
  double shift_x = (-1 + (2 * index_x + 1) / N_x);
  double shift_y = (-1 + (2 * index_y + 1) / N_y);
  double shift_z = (-1 + (2 * index_z + 1) / N_z);
  DataVector block_logical_x = 1 / N_x * new_coords.get(0) + shift_x;
  DataVector block_logical_y = 1 / N_y * new_coords.get(1) + shift_y;
  DataVector block_logical_z = 1 / N_z * new_coords.get(2) + shift_z;
  std::cout << element_id << '\n';
  tnsr::I<DataVector, 3, Frame::BlockLogical> block_logical_coords{
      block_logical_x.size(), 0.};
  block_logical_coords.get(0) = block_logical_x;
  block_logical_coords.get(1) = block_logical_y;
  block_logical_coords.get(2) = block_logical_z;
  std::cout << block_logical_coords << '\n';

  return block_logical_coords;
}

// Functions that compute the connectivity within the block
std::map<std::tuple<double, double, double>, size_t> generate_grid_point_map(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& coord_data) {
  std::map<std::tuple<double, double, double>, size_t> grid_point_map;
  DataVector coord_data_x = coord_data.get(0);
  DataVector coord_data_y = coord_data.get(1);
  DataVector coord_data_z = coord_data.get(2);
  for (size_t i = 0; i < coord_data_x.size(); ++i) {
    std::tuple<double, double, double> coord_data_point;
    std::get<0>(coord_data_point) = coord_data_x[i];
    std::get<1>(coord_data_point) = coord_data_y[i];
    std::get<2>(coord_data_point) = coord_data_z[i];
    grid_point_map.insert(std::pair<std::tuple<double, double, double>, size_t>(
        coord_data_point, i));
  }
  std::cout << "Finished generate_grid_point_map" << '\n';
  return grid_point_map;
}

std::vector<double> sort_and_order(DataVector& data_vector) {
  std::vector<double> ordered_coords;
  std::vector<double> data_list;
  for (size_t i = 0; i < data_vector.size(); ++i) {
    data_list.emplace_back(data_vector[i]);
  }
  sort(data_list.begin(), data_list.end());
  ordered_coords.push_back(data_list[0]);
  for (size_t i = 1; i < data_list.size(); ++i) {
    if (data_list[i] == ordered_coords.end()[-1]) {
      continue;
    } else {
      ordered_coords.push_back(data_list[i]);
    }
  }
  std::cout << "Finished sort_and_order" << '\n';
  return ordered_coords;
}

std::vector<std::tuple<double, double, double>> build_connectivity_by_element(
    std::vector<double>& sorted_x, std::vector<double>& sorted_y,
    std::vector<double>& sorted_z) {
  std::vector<std::tuple<double, double, double>> connectivity_as_tuples;
  for (size_t k = 0; k < sorted_z.size() - 1; ++k) {
    for (size_t j = 0; j < sorted_y.size() - 1; ++j) {
      for (size_t i = 0; i < sorted_x.size() - 1; ++i) {
        connectivity_as_tuples.insert(
            connectivity_as_tuples.end(),
            {std::make_tuple(sorted_x[i], sorted_y[j], sorted_z[k]),
             std::make_tuple(sorted_x[i + 1], sorted_y[j], sorted_z[k]),
             std::make_tuple(sorted_x[i + 1], sorted_y[j + 1], sorted_z[k]),
             std::make_tuple(sorted_x[i], sorted_y[j + 1], sorted_z[k]),
             std::make_tuple(sorted_x[i], sorted_y[j], sorted_z[k + 1]),
             std::make_tuple(sorted_x[i + 1], sorted_y[j], sorted_z[k + 1]),
             std::make_tuple(sorted_x[i + 1], sorted_y[j + 1], sorted_z[k + 1]),
             std::make_tuple(sorted_x[i], sorted_y[j + 1], sorted_z[k + 1])});
      }
    }
  }
  std::cout << "Finished build_connectivity_by_element" << '\n';
  return connectivity_as_tuples;
}

std::vector<double> generate_new_connectivity(
    tnsr::I<DataVector, 3, Frame::ElementLogical>& logical_coords) {
  std::map<std::tuple<double, double, double>, size_t> grid_point_map =
      generate_grid_point_map(logical_coords);
  DataVector logical_coords_x = logical_coords.get(0);
  DataVector logical_coords_y = logical_coords.get(1);
  DataVector logical_coords_z = logical_coords.get(2);
  std::vector<double> ordered_x = sort_and_order(logical_coords_x);
  std::vector<double> ordered_y = sort_and_order(logical_coords_y);
  std::vector<double> ordered_z = sort_and_order(logical_coords_z);
  std::vector<std::tuple<double, double, double>> connectivity_of_tuples =
      build_connectivity_by_element(ordered_x, ordered_y, ordered_z);
  std::vector<double> connectivity;
  for (const std::tuple<double, double, double>& it : connectivity_of_tuples) {
    connectivity.push_back(grid_point_map[it]);
  }
  // std::cout << connectivity << '\n';
  std::cout << "Finished generate_new_connectivity" << '\n';
  return connectivity;
}

void block_connectivity(const std::string& file_name,
                        const std::string& subfile_name) {
  h5::H5File<h5::AccessType::ReadWrite> data_file(file_name, true);
  const auto& volume_file = data_file.get<h5::VolumeData>("/" + subfile_name);
  auto observation_ids = volume_file.list_observation_ids();
  const auto& single_id =
      observation_ids[0];  // Hard coded for only 1 observation ID
  std::vector<double> new_connectivity;
  for (size_t i = 0; i < volume_file.get_bases(single_id).size(); ++i) {
    auto new_mesh = generate_mesh(volume_file, single_id, i);
    auto new_coords = logical_coordinates(std::get<0>(new_mesh));
    std::cout << new_coords << '\n';
    std::cout << block_logical_coords(new_coords, std::get<1>(new_mesh))
              << '\n';
    auto generated_connectivity = generate_new_connectivity(new_coords);
    new_connectivity.insert(new_connectivity.end(),
                            generated_connectivity.begin(),
                            generated_connectivity.end());
  }
  // std::cout << new_connectivity << '\n';
}

// def extend_connectivity(input_file):

//     if sys.version_info < (3, 0):
//         logging.warning("You are attempting to run this script with "
//                         "python 2, which is deprecated. GenerateXdmf.py might
//                         " "hang or run very slowly using python 2. Please use
//                         " "python 3 instead.")

//     with h5py.File(input_file, 'r+') as dataset:
//         subfile_keys = list(dataset.keys())
//         subfile_name = subfile_keys[0]
//         data = dataset[subfile_name]
//         observation_keys = list(data.keys())
//         for i in range(len(observation_keys)):
//             print("Starting observation key number " + str(i))  ############
//             data_y = np.array(
//                 dataset[subfile_name + "/" + observation_keys[i] +
//                         "/InertialCoordinates_y"])
//             data_x = np.array(
//                 dataset[subfile_name + "/" + observation_keys[i] +
//                         "/InertialCoordinates_x"])
//             data_z = np.array(
//                 dataset[subfile_name + "/" + observation_keys[i] +
//                         "/InertialCoordinates_z"])
//             new_connect = generate_new_connectivity(data_x, data_y, data_z)
//             del dataset[subfile_name + "/" + observation_keys[i] +
//                         "/connectivity"]
//             dataset.create_dataset(subfile_name + "/" + observation_keys[i] +
//                                    "/connectivity",
//                                    data=new_connect)
//         dataset.close()

int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;

  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "file_name", boost::program_options::value<std::string>()->required(),
      "name of the file")(
      "subfile_name", boost::program_options::value<std::string>()->required(),
      "subfile name of the volume file in the H5 file (omit file "
      "extension)");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("file_name") == 0u or
      vars.count("subfile_name") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  block_connectivity(vars["file_name"].as<std::string>(),
                     vars["subfile_name"].as<std::string>());
}
