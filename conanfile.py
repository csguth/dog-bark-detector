from conans import ConanFile, CMake, tools


class DogBarkDetectorConan(ConanFile):
    name = "dog-bark-detector"
    version = "0.1"
    author = "Chrystian Guth <csguth@gmail.com>"
    url = "https://github.com/csguth/dog-bark-detector"
    description = "Detect dog bark from CNN based spectrogram classification (forked from lincolnhard/dog-bark-detector)"
    topics = ("cnn", "ml", "dog")
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake_find_package"
    exports_sources = "**"

    def requirements(self):
        self.requires("libpng/1.6.37")
        self.requires("zlib/1.2.12")
        self.requires("opencv/2.4.13.7")
        self.requires("darknet/cci.20180914")
        self.requires("cairo/[^1.17]")
        self.requires("fftw/[^3.3]")
        self.requires("libsndfile/[^1.0]")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        pass

    def package_info(self):
        pass

