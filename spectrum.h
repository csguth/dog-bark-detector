#pragma once

#include <functional>
#include <memory>
#include <optional>

class Spectrum final {
public:
    using window_function_t = std::function<void(double*, size_t)>;
    
    ~Spectrum() noexcept;
    Spectrum(Spectrum&&);
    Spectrum(const Spectrum&) = delete;
    Spectrum& operator=(Spectrum&&);
    Spectrum& operator=(const Spectrum&) = delete;
    friend void swap(Spectrum&, Spectrum&);
    
    static std::optional<Spectrum> create(int speclen, double* timeDomain, double* freqDomain) noexcept;

    void executeFft();
    
    void applyWindow(double* data, size_t dataLen) const;
    
private:
    Spectrum();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit Spectrum(std::unique_ptr<Impl> impl);

};
