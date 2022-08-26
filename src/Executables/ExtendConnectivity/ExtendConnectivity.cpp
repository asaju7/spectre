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
#include "IO/H5/Helpers.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

void extend_connectivity(const std::string& file_name,
                         const std::string& subfile_name,
                         const bool& print_size) {
  h5::H5File<h5::AccessType::ReadWrite> data_file(file_name, true);
  auto& volume_file = data_file.get<h5::VolumeData>("/" + subfile_name);
  const std::vector<size_t>& observation_ids =
      volume_file.list_observation_ids();

  volume_file.write_new_connectivity_data(observation_ids, print_size);
}

/*
 * This executable is used for extending the connectivity inside of a single
 * HDF5 volume file for some SpECTRE evolution, in order to fill in gaps between
 * elements.
 */
int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;

  boost::program_options::options_description desc("Options");
  desc.add_options()(
      "help,h,",
      "show this help message\n\nNote: This does not work with subcell systems "
      "or 1D/2D systems, and the connectivity only extends *within* each block "
      "and not between them. Further, you need to manually edit each "
      "ObservationID in the generated XDMF to properly account for the new "
      "number of elements that have been created\n")(
      "file_name", boost::program_options::value<std::string>()->required(),
      "name of the file")(
      "subfile_name", boost::program_options::value<std::string>()->required(),
      "subfile name of the volume file in the H5 file (omit file "
      "extension)")(
      "print_size", boost::program_options::value<bool>()->default_value(false),
      "true if the length of the extended connectivity is to be printed, false "
      "otherwise (optional, default is false if unspecified)");

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

  extend_connectivity(vars["file_name"].as<std::string>(),
                      vars["subfile_name"].as<std::string>(),
                      vars["print_size"].as<bool>());
}
