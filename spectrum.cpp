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

struct spectrum final
{
    ~spectrum()
    {
        fftw_destroy_plan (plan) ;
    }
    
    int speclen ;
    enum WINDOW_FUNCTION wfunc ;
    fftw_plan plan ;

    std::vector<double> time_domain ;
    std::vector<double> window ;
    std::vector<double> freq_domain ;
    std::vector<double> mag_spec ;

    double data [] ;
} ;


spectrum *
create_spectrum (int speclen, enum WINDOW_FUNCTION window_function)
{
    try
    {
        auto spec = std::make_unique<spectrum>();
        spec->wfunc = window_function ;
        spec->speclen = speclen ;

        /* mag_spec has values from [0..speclen] inclusive for 0Hz to Nyquist.
        ** time_domain has an extra element to be able to interpolate between
        ** samples for better time precision, hoping to eliminate artifacts.
        */
        spec->time_domain.resize(2*speclen+1);
        spec->window.resize(2*speclen);
        spec->freq_domain.resize(2*speclen);
        spec->mag_spec.resize(speclen+1);

        spec->plan = fftw_plan_r2r_1d (2 * speclen, spec->time_domain.data(), spec->freq_domain.data(), FFTW_R2HC, FFTW_MEASURE | FFTW_PRESERVE_INPUT) ;
        if (spec->plan == nullptr)
        {
            printf ("%s:%d : fftw create plan failed.\n", __func__, __LINE__) ;
            return {};
        } ;

        switch (spec->wfunc)
        {	case RECTANGULAR :
                break ;
            case KAISER :
                calc_kaiser_window (spec->window.data(), 2 * speclen, 20.0) ;
                break ;
            case NUTTALL:
                calc_nuttall_window (spec->window.data(), 2 * speclen) ;
                break ;
            case HANN :
                calc_hann_window (spec->window.data(), 2 * speclen) ;
                break ;
            default :
                printf ("Internal error: Unknown window_function.\n") ;
                return {};
            } ;

        return spec.release() ;
    } catch (...) {
        return {};
    }
} /* create_spectrum */


void destroy_spectrum (spectrum * spec)
{
    delete spec;
}

double calc_magnitude_spectrum (spectrum * spec)
{
	double max ;
	int k, freqlen ;

	freqlen = 2 * spec->speclen ;

	if (spec->wfunc != RECTANGULAR)
		for (k = 0 ; k < 2 * spec->speclen ; k++)
			spec->time_domain [k] *= spec->window [k] ;


	fftw_execute (spec->plan) ;

	/* Convert from FFTW's "half complex" format to an array of magnitudes.
	** In HC format, the values are stored:
	** r0, r1, r2 ... r(n/2), i(n+1)/2-1 .. i2, i1
	**/
	max = spec->mag_spec [0] = fabs (spec->freq_domain [0]) ;

	for (k = 1 ; k < spec->speclen ; k++)
	{	double re = spec->freq_domain [k] ;
		double im = spec->freq_domain [freqlen - k] ;
		spec->mag_spec [k] = sqrt (re * re + im * im) ;
		max = MAX (max, spec->mag_spec [k]) ;
		} ;
	/* Lastly add the point for the Nyquist frequency */
	spec->mag_spec [spec->speclen] = fabs (spec->freq_domain [spec->speclen]) ;

	return max ;
} /* calc_magnitude_spectrum */

double * spectrum_time_domain(spectrum* self)
{
    return self->time_domain.data();
}

double * spectrum_mag_spec(spectrum* self)
{
    return self->mag_spec.data();
}
