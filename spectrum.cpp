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
    ~Impl()
    {
        fftw_destroy_plan (plan) ;
    }
    
    int speclen ;
    WindowFunction wfunc ;
    fftw_plan plan ;
    
    std::vector<double> time_domain ;
    std::vector<double> window ;
    std::vector<double> freq_domain ;
    std::vector<double> mag_spec ;
    
    double data [] ;
};

std::optional<Spectrum> Spectrum::create(int speclen, WindowFunction window_function) noexcept {
    try {
        auto spec = Spectrum{};
        spec.impl->wfunc = window_function ;
        spec.impl->speclen = speclen ;
        
        /* mag_spec has values from [0..speclen] inclusive for 0Hz to Nyquist.
         ** time_domain has an extra element to be able to interpolate between
         ** samples for better time precision, hoping to eliminate artifacts.
         */
        spec.impl->time_domain.resize(2*speclen+1);
        spec.impl->window.resize(2*speclen);
        spec.impl->freq_domain.resize(2*speclen);
        spec.impl->mag_spec.resize(speclen+1);
        
        spec.impl->plan = fftw_plan_r2r_1d (2 * speclen, spec.impl->time_domain.data(), spec.impl->freq_domain.data(), FFTW_R2HC, FFTW_MEASURE | FFTW_PRESERVE_INPUT) ;
        if (spec.impl->plan == nullptr)
        {
            printf ("%s:%d : fftw create plan failed.\n", __func__, __LINE__) ;
            return {};
        } ;
        
        switch (spec.impl->wfunc)
        {    case WindowFunction::RECTANGULAR :
                break ;
            case WindowFunction::KAISER :
                calc_kaiser_window (spec.impl->window.data(), 2 * speclen, 20.0) ;
                break ;
            case WindowFunction::NUTTALL:
                calc_nuttall_window (spec.impl->window.data(), 2 * speclen) ;
                break ;
            case WindowFunction::HANN :
                calc_hann_window (spec.impl->window.data(), 2 * speclen) ;
                break ;
            default :
                printf ("Internal error: Unknown window_function.\n") ;
                return {};
        } ;
        
        return spec;
    } catch (...) {
        return {};
    }
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

Spectrum::Spectrum()
: impl{ std::make_unique<Impl>() }
{}

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

    if (impl->wfunc != WindowFunction::RECTANGULAR)
        for (k = 0 ; k < 2 * impl->speclen ; k++)
            impl->time_domain [k] *= impl->window [k] ;


    fftw_execute (impl->plan) ;

    /* Convert from FFTW's "half complex" format to an array of magnitudes.
    ** In HC format, the values are stored:
    ** r0, r1, r2 ... r(n/2), i(n+1)/2-1 .. i2, i1
    **/
    max = impl->mag_spec [0] = fabs (impl->freq_domain [0]) ;

    for (k = 1 ; k < impl->speclen ; k++)
    {    double re = impl->freq_domain [k] ;
        double im = impl->freq_domain [freqlen - k] ;
        impl->mag_spec [k] = sqrt (re * re + im * im) ;
        max = MAX (max, impl->mag_spec [k]) ;
        } ;
    /* Lastly add the point for the Nyquist frequency */
    impl->mag_spec [impl->speclen] = fabs (impl->freq_domain [impl->speclen]) ;

    return max ;
}
