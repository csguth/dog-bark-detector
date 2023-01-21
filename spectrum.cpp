#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>

#include <fftw3.h>

#include <sndfile.h>

#include "common.h"
#include "window.h"
#include "spectrum.h"

#include <memory>
#include <vector>

namespace {

    std::vector<double> make_kaiser_window(size_t size)
    {
        auto data = std::vector<double>(size);
        calc_kaiser_window(data.data(), data.size(), 20.0);
        return data;
    }

}

struct Spectrum::Impl {
    
    Impl(int speclen, double* timeDomain, double* freqDomain)
        : speclen{speclen}
        , window(make_kaiser_window(2*speclen))
        , mag_spec(speclen+1)
        , plan{ nullptr }
    {
        auto const plan = fftw_plan_r2r_1d (2 * speclen, timeDomain, freqDomain, FFTW_R2HC, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
        if (plan == nullptr)
        {
            throw std::runtime_error{"failed to create fftw"};
        }
        this->plan = plan;
    }
    
    ~Impl()
    {
        if (plan != nullptr)
        {
            fftw_destroy_plan(plan);
        }
    }
    
    const int speclen;
    const window_function_t windowFunction;
    const std::vector<double> window;
    std::vector<double> mag_spec;
    
    fftw_plan plan;
};

std::optional<Spectrum> Spectrum::create(int speclen, double* timeDomain, double* freqDomain) noexcept {
    try {
        return Spectrum{
            std::make_unique<Impl>(speclen, timeDomain, freqDomain)
        };
    } catch (...) {}
    return {};
}

void swap(Spectrum& lhs, Spectrum& rhs)
{
    std::swap(lhs.impl, rhs.impl);
}

Spectrum::Spectrum(Spectrum&& rhs)
: Spectrum{}
{
    swap(*this, rhs);
}

Spectrum& Spectrum::operator=(Spectrum&& rhs)
{
    swap(*this, rhs);
    return *this;
}
Spectrum::~Spectrum() noexcept = default;

Spectrum::Spectrum() = default;

void Spectrum::applyWindow(double* timeDomain, size_t dataLen) const
{
    for (size_t k = 0 ; k < 2 * impl->speclen ; k++)
    {
        timeDomain[k] *= impl->window[k];
    }
}


void Spectrum::executeFft()
{
    fftw_execute(impl->plan);
}

Spectrum::Spectrum(std::unique_ptr<Impl> impl)
: impl{std::move(impl)} {
}

