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

struct Spectrum::Impl {
    
    Impl(int speclen, window_function_t windowFunction)
        : speclen{speclen}
        , windowFunction{std::move(windowFunction)}
        , time_domain(2*speclen+1)
        , window(2*speclen)
        , freq_domain(2*speclen)
        , mag_spec(speclen+1)
        , plan{ nullptr }
    {
        auto const plan = fftw_plan_r2r_1d (2 * speclen, time_domain.data(), freq_domain.data(), FFTW_R2HC, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
        if (plan == nullptr)
        {
            throw std::runtime_error{"failed to create fftw"};
        }
        
        if (!this->windowFunction)
        {
            throw std::runtime_error{"invalid window function"};
        }
        this->plan = plan;
        this->windowFunction(window.data(), window.size());
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
    
    std::vector<double> time_domain;
    std::vector<double> window;
    std::vector<double> freq_domain;
    std::vector<double> mag_spec;
    
    fftw_plan plan;
};

std::optional<Spectrum> Spectrum::create(int speclen, window_function_t windowFunction) noexcept {
    try {
        return Spectrum{
            std::make_unique<Impl>(speclen, std::move(windowFunction))
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

double* Spectrum::magSpec()
{
    return impl->mag_spec.data();
}
double* Spectrum::timeDomain()
{
    return impl->time_domain.data();
}
double Spectrum::calcMagnitudeSpectrum()
{
    double max ;
    int k, freqlen ;

    freqlen = 2 * impl->speclen ;

//    if (impl->wfunc != WindowFunction::RECTANGULAR)
        for (k = 0 ; k < 2 * impl->speclen ; k++)
            impl->time_domain [k] *= impl->window [k] ;


    fftw_execute (impl->plan) ;

    /* Convert from FFTW's "half complex" format to an array of magnitudes.
    ** In HC format, the values are stored:
    ** r0, r1, r2 ... r(n/2), i(n+1)/2-1 .. i2, i1
    **/
    max = impl->mag_spec [0] = fabs (impl->freq_domain [0]) ;

    for (k = 1 ; k < impl->speclen ; k++)
    {
        double re = impl->freq_domain [k] ;
        double im = impl->freq_domain [freqlen - k] ;
        impl->mag_spec [k] = sqrt (re * re + im * im) ;
        max = MAX (max, impl->mag_spec [k]) ;
    } ;
    /* Lastly add the point for the Nyquist frequency */
    impl->mag_spec [impl->speclen] = fabs (impl->freq_domain [impl->speclen]) ;

    return max ;
}

Spectrum::Spectrum(std::unique_ptr<Impl> impl)
: impl{std::move(impl)} {
}

