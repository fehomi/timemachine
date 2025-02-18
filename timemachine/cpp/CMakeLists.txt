cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_PYTHON_VERSION 3.7 CACHE STRING "Which version of python we're building wrappers against")

project(timemachine LANGUAGES CXX CUDA)

find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

string(APPEND CMAKE_CUDA_FLAGS "-O3 -lineinfo")
message(${CMAKE_CUDA_FLAGS})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set(CMAKE_INSTALL_PREFIX ${PARENT_DIR} CACHE PATH "Default Install Path" FORCE)
endif()

set(LIBRARY_NAME custom_ops)
set(PYBIND_SRC_DIR pybind11)

if(NOT EXISTS ${PYBIND_SRC_DIR})
  execute_process(COMMAND git clone --branch v2.6 https://github.com/pybind/pybind11.git ${PYBIND_SRC_DIR})
endif()

set(EIGEN_SRC_DIR eigen)
if(NOT EXISTS ${EIGEN_SRC_DIR})
  execute_process(COMMAND git clone --branch 3.3.9 https://gitlab.com/libeigen/eigen.git ${EIGEN_SRC_DIR})
endif()

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_SRC_DIR})


# NO_EXTRAS is needed since cuda doesn't use flto
pybind11_add_module(${LIBRARY_NAME} SHARED NO_EXTRAS
  src/fixed_point.hpp
  src/wrap_kernels.cpp
  src/potential.cu
  src/bound_potential.cu
  src/centroid_restraint.cu
  src/gpu_utils.cu
  src/vendored/hilbert.cpp
  src/nonbonded.cu
  src/nonbonded_all_pairs.cu
  src/nonbonded_pair_list.cu
  src/nonbonded_interaction_group.cu
  src/neighborlist.cu
  src/harmonic_bond.cu
  src/harmonic_angle.cu
  src/periodic_torsion.cu
  src/integrator.cu
  src/context.cu
  src/barostat.cu
  src/rmsd_align.cpp
  src/summed_potential.cu
  src/device_buffer.cu
  src/kernels/k_nonbonded.cu
  src/kernels/nonbonded_common.cu
)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(src/kernels)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/eigen)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/${CUB_SRC_DIR})

set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 17)

if (NOT CUDA_ARCH)
  set(CUDA_ARCH "75")
endif()
set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH}")
unset(CUDA_ARCH CACHE)

target_link_libraries(${LIBRARY_NAME} PRIVATE -lcurand -lcuda -lcudart -lcudadevrt -lnvrtc)
set_target_properties(${LIBRARY_NAME} PROPERTIES PREFIX "")

install(TARGETS ${LIBRARY_NAME} DESTINATION "lib")
