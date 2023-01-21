#pragma once

#include <memory>
#include <optional>

class Spectrum final {
public:
    ~Spectrum() noexcept;
    Spectrum(Spectrum&&);
    Spectrum(const Spectrum&) = delete;
    Spectrum& operator=(Spectrum&&);
    Spectrum& operator=(const Spectrum&) = delete;
    friend void swap(Spectrum&, Spectrum&);
    
    static std::optional<Spectrum> create(int speclen, WindowFunction window_function) noexcept;
    
    double* magSpec();
    double* timeDomain();
    double calcMagnitudeSpectrum();
    
private:
    Spectrum();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl;

};
