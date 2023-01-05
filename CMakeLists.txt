cmake_minimum_required(VERSION 3.19.2)
project(dog-bark-detector)

find_package(SndFile REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(cairo REQUIRED)
find_package(darknet REQUIRED)
find_package(OpenCV REQUIRED)

add_library(
    dbd
    common.c
    create_spectrogram.c
    spectrum.c
    window.c
)

add_library(
    dbdpp
    network.cpp
    network.hpp
)
set_property(TARGET dbdpp PROPERTY CXX_STANDARD 20)


target_link_libraries(dbdpp
    PRIVATE
    darknet::darknet
)

target_link_libraries(
    dbd
    PUBLIC
    SndFile::SndFile
    FFTW3::FFTW3
    cairo::cairo
    darknet::darknet
)

add_executable(
    classification_from_file
    classification_from_file.cpp
)

target_link_libraries(
    classification_from_file
    PRIVATE
    dbd
    dbdpp
    opencv::opencv
)

set_property(TARGET classification_from_file PROPERTY CXX_STANDARD 20)
