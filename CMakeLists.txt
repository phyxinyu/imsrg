########################################################################################################################
### IMSRG LIBRARY MAIN
########################################################################################################################

cmake_minimum_required(VERSION 3.22)

########################################################################################################################
### PROJECT DESCRIPTION AND META INFO
########################################################################################################################

set(IMSRG_PROJECT_NAME        "imsrg++")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
option(IMSRG_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(USE_ARMADILLO_EXTERN "Use version of armadillo in extern" ON)
option(IMSRG_USE_MPI "Build optional MPI support for scalar IMSRG(2)" OFF)
set(IMSRG_PLATFORM "" CACHE STRING "Dependency platform preset. Empty means auto-detect; valid values: linux, macos")
set_property(CACHE IMSRG_PLATFORM PROPERTY STRINGS "" linux macos)
########################################################################################################################
### DECLARE PROJECT
########################################################################################################################

project(imsrg++ VERSION 0.1.0 LANGUAGES CXX)

set(IMSRG_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

if(NOT IMSRG_PLATFORM)
    if(APPLE)
        set(IMSRG_PLATFORM "macos" CACHE STRING "Dependency platform preset. Empty means auto-detect; valid values: linux, macos" FORCE)
    else()
        set(IMSRG_PLATFORM "linux" CACHE STRING "Dependency platform preset. Empty means auto-detect; valid values: linux, macos" FORCE)
    endif()
endif()

if(IMSRG_PLATFORM STREQUAL "macos")
    set(IMSRG_HOMEBREW_PREFIX "" CACHE PATH "Homebrew prefix for macOS dependencies")
    if(NOT IMSRG_HOMEBREW_PREFIX)
        if(EXISTS "/opt/homebrew")
            set(IMSRG_HOMEBREW_PREFIX "/opt/homebrew" CACHE PATH "Homebrew prefix for macOS dependencies" FORCE)
        elseif(EXISTS "/usr/local")
            set(IMSRG_HOMEBREW_PREFIX "/usr/local" CACHE PATH "Homebrew prefix for macOS dependencies" FORCE)
        endif()
    endif()

    foreach(IMSRG_BREW_PKG libomp openblas gsl boost zlib)
        if(EXISTS "${IMSRG_HOMEBREW_PREFIX}/opt/${IMSRG_BREW_PKG}")
            list(PREPEND CMAKE_PREFIX_PATH "${IMSRG_HOMEBREW_PREFIX}/opt/${IMSRG_BREW_PKG}")
        endif()
    endforeach()

    if(EXISTS "${IMSRG_HOMEBREW_PREFIX}/opt/libomp/include/omp.h" AND EXISTS "${IMSRG_HOMEBREW_PREFIX}/opt/libomp/lib/libomp.dylib")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${IMSRG_HOMEBREW_PREFIX}/opt/libomp/include" CACHE STRING "OpenMP CXX flags")
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP CXX library names")
        set(OpenMP_omp_LIBRARY "${IMSRG_HOMEBREW_PREFIX}/opt/libomp/lib/libomp.dylib" CACHE FILEPATH "OpenMP libomp library")
    endif()

    set(BLA_VENDOR OpenBLAS CACHE STRING "BLAS vendor" FORCE)
elseif(IMSRG_PLATFORM STREQUAL "linux")
    set(IMSRG_LOCAL_PREFIX "/tns/xuxinyu/usr" CACHE PATH "Linux dependency prefix")
    if(EXISTS "${IMSRG_LOCAL_PREFIX}")
        list(PREPEND CMAKE_PREFIX_PATH "${IMSRG_LOCAL_PREFIX}")
        list(PREPEND CMAKE_INCLUDE_PATH "${IMSRG_LOCAL_PREFIX}/include")
        list(PREPEND CMAKE_LIBRARY_PATH "${IMSRG_LOCAL_PREFIX}/lib")
        include_directories(SYSTEM "${IMSRG_LOCAL_PREFIX}/include")
        link_directories("${IMSRG_LOCAL_PREFIX}/lib")
    endif()

    if(EXISTS "${IMSRG_LOCAL_PREFIX}/lib/libgsl.so")
        set(GSL_LIBRARY "${IMSRG_LOCAL_PREFIX}/lib/libgsl.so" CACHE FILEPATH "GSL library" FORCE)
    endif()
    if(EXISTS "${IMSRG_LOCAL_PREFIX}/lib/libgslcblas.so")
        set(GSL_CBLAS_LIBRARY "${IMSRG_LOCAL_PREFIX}/lib/libgslcblas.so" CACHE FILEPATH "GSL CBLAS library" FORCE)
    endif()
    if(EXISTS "${IMSRG_LOCAL_PREFIX}/include/gsl/gsl_version.h")
        set(GSL_INCLUDE_DIR "${IMSRG_LOCAL_PREFIX}/include" CACHE PATH "GSL include directory" FORCE)
    endif()

    set(BLA_VENDOR Generic CACHE STRING "BLAS vendor" FORCE)
else()
    message(FATAL_ERROR "Unsupported IMSRG_PLATFORM='${IMSRG_PLATFORM}'. Use linux or macos.")
endif()

message(STATUS "IMSRG dependency platform: ${IMSRG_PLATFORM}")

find_package(OpenMP REQUIRED COMPONENTS CXX)
if(IMSRG_USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()
find_package(BLAS)
if(NOT BLAS_FOUND AND IMSRG_PLATFORM STREQUAL "linux")
    set(BLAS_LIBRARIES
        "-L${IMSRG_LOCAL_PREFIX}/lib"
        "-lz"
        "-lgslcblas"
        "-lgsl"
        "-lblas"
        "-lcblas"
        "-llapack"
        CACHE STRING "Linux fallback BLAS/LAPACK link flags" FORCE
    )
    set(BLAS_FOUND TRUE)
    message(STATUS "Using Linux fallback BLAS/LAPACK link flags: ${BLAS_LIBRARIES}")
endif()
if(NOT BLAS_FOUND)
    message(FATAL_ERROR "BLAS was not found for IMSRG_PLATFORM='${IMSRG_PLATFORM}'. On macOS, install openblas with Homebrew or set CMAKE_PREFIX_PATH/BLAS_LIBRARIES.")
endif()
if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
endif()

find_package(LAPACK)
if(NOT LAPACK_FOUND AND IMSRG_PLATFORM STREQUAL "linux")
    set(LAPACK_LIBRARIES
        "-L${IMSRG_LOCAL_PREFIX}/lib"
        "-llapack"
        CACHE STRING "Linux fallback LAPACK link flags" FORCE
    )
    set(LAPACK_FOUND TRUE)
    message(STATUS "Using Linux fallback LAPACK link flags: ${LAPACK_LIBRARIES}")
endif()
if(NOT LAPACK_FOUND)
    message(FATAL_ERROR "LAPACK was not found for IMSRG_PLATFORM='${IMSRG_PLATFORM}'. On macOS, install openblas with Homebrew or set CMAKE_PREFIX_PATH/LAPACK_LIBRARIES.")
endif()
if(NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
    set_target_properties(LAPACK::LAPACK PROPERTIES INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}")
endif()

find_package(GSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Boost REQUIRED)
if (NOT USE_ARMADILLO_EXTERN)
    find_package(Armadillo)
endif()

set(IMSRG_PLATFORM_CONFIGURED TRUE CACHE INTERNAL "Top-level platform dependencies have been configured")

if(IMSRG_BUILD_PYTHON_BINDINGS)
    # We only need this for the Python bindings.
    # In principle, everything should be faster if we could avoid this.
    set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

add_subdirectory(src)
if(IMSRG_BUILD_PYTHON_BINDINGS)
    #ADD tests to be run directly in the build process
    enable_testing()
    if(NOT PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE python3)
    endif()

    function(add_pyimsrg_test test_name script_path)
        add_test(
            NAME ${test_name}
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:pyIMSRG>"
                    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script_path}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    endfunction()

    add_pyimsrg_test(UnitTest tests/UnitTest.py)
    add_pyimsrg_test(UnitTestCommutators2 test/UnitTestCommutators2.py)
    add_pyimsrg_test(UnitTestCommutators3 test/UnitTestCommutators3.py)
    add_pyimsrg_test(UnitTestFactorized test/UnitTestFactorized.py)
    add_pyimsrg_test(UnitTestTriples test/UnitTestTriples.py)
    add_pyimsrg_test(UnitTestTensor test/UnitTestTensor.py)
    add_pyimsrg_test(UnitTestNormalOrdering test/UnitTestNormalOrdering.py)
    add_pyimsrg_test(UnitTestEventIMSRG2 test/UnitTestEventIMSRG2.py)
endif()
