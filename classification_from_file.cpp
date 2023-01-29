#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>

#include <fftw3.h>
#include <sndfile.hh>

#include "window.h"
#include "common.h"
#include "spectrum.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "network.hpp"


static void
get_colour_map_value (float value, double spec_floor_db, unsigned char colour [3])
{	static unsigned char map [][3] =
    {	/* These values were originally calculated for a dynamic range of 180dB. */
        {	255,	255,	255	},	/* -0dB */
        {	240,	254,	216	},	/* -10dB */
        {	242,	251,	185	},	/* -20dB */
        {	253,	245,	143	},	/* -30dB */
        {	253,	200,	102	},	/* -40dB */
        {	252,	144,	66	},	/* -50dB */
        {	252,	75,		32	},	/* -60dB */
        {	237,	28,		41	},	/* -70dB */
        {	214,	3,		64	},	/* -80dB */
        {	183,	3,		101	},	/* -90dB */
        {	157,	3,		122	},	/* -100dB */
        {	122,	3,		126	},	/* -110dB */
        {	80,		2,		110	},	/* -120dB */
        {	45,		2,		89	},	/* -130dB */
        {	19,		2,		70	},	/* -140dB */
        {	1,		3,		53	},	/* -150dB */
        {	1,		3,		37	},	/* -160dB */
        {	1,		2,		19	},	/* -170dB */
        {	0,		0,		0	},	/* -180dB */
    } ;
    
    float rem ;
    int indx ;
    
    if (value >= 0.0)
    {	colour [0] = colour [1] = colour [2] = 255 ;
        return ;
    } ;
    
    value = fabs (value * (-180.0 / spec_floor_db) * 0.1) ;
    
    indx = lrintf (floor (value)) ;
    
    if (indx < 0)
    {	printf ("\nError : colour map array index is %d\n\n", indx) ;
        exit (1) ;
    } ;
    
    if (indx >= ARRAY_LEN (map) - 1)
    {	colour [0] = colour [1] = colour [2] = 0 ;
        return ;
    } ;
    
    rem = fmod (value, 1.0) ;
    
    colour [0] = lrintf ((1.0 - rem) * map [indx][0] + rem * map [indx + 1][0]) ;
    colour [1] = lrintf ((1.0 - rem) * map [indx][1] + rem * map [indx + 1][1]) ;
    colour [2] = lrintf ((1.0 - rem) * map [indx][2] + rem * map [indx + 1][2]) ;
    
    return ;
}

/* The greatest number of linear ticks seems to occurs from 0-14000 (15 ticks).
 ** The greatest number of log ticks occurs 10-99999 or 11-100000 (35 ticks).
 ** Search for "worst case" for the commentary below that says why it is 35.
 */
typedef struct
{	double value [40] ;  /* 35 or more */
    double distance [40] ;
    /* The digit that changes from label to label.
     ** This ensures that a range from 999 to 1001 prints 999.5 and 1000.5
     ** instead of 999 1000 1000 1000 1001.
     */
    int decimal_places_to_print ;
} TICKS ;

/* Decide where to put ticks and numbers on an axis.
 **
 ** Graph-labelling convention is that the least significant digit that changes
 ** from one label to the next should change by 1, 2 or 5, so we step by the
 ** largest suitable value of 10^n * {1, 2 or 5} that gives us the required
 ** number of divisions / numeric labels.
 */

/* The old code used to make 6 to 14 divisions and number every other tick.
 ** What we now mean by "division" is one of teh gaps between numbered segments
 ** so we ask for a minimum of 3 to give the same effect as the old minimum of
 ** 6 half-divisions.
 ** This results in the same axis labelling for all maximum values
 ** from 0 to 12000 in steps of 1000 and gives sensible results from 13000 on,
 ** to a maximum of 7 divisions and 8 labels from 0 to 14000.
 **/
#define TARGET_DIVISIONS 3

/* Value to store in the ticks.value[k] field to mean
 ** "Put a tick here, but don't print a number."
 ** NaN (0.0/0.0) is untestable without isnan() so use a random value.
 */
#define NO_NUMBER (M_PI)		/* They're unlikely to hit that! */

/* Is this entry in "ticks" one of the numberless ticks? */
#define JUST_A_TICK(ticks, k)	(ticks.value [k] == NO_NUMBER)

/* A tolerance to use in floating point < > <= >= comparisons so that
 ** imprecision doesn't prevent us from printing an initial or final label
 ** if it should fall exactly on min or max but doesn't due to FP problems.
 ** For example, for 0-24000, the calculations might give 23999.9999999999.
 */
#define DELTA (1e-10)

static int	/* Forward declaration */
calculate_log_ticks (double min, double max, double distance, TICKS * ticks) ;

/* log_scale is pseudo-boolean:
 ** 0 means use a linear scale,
 ** 1 means use a log scale and
 ** 2 is an internal value used when calling back from calculate_log_ticks() to
 **   label the range with linear numbering but logarithmic spacing.
 */

static int
calculate_ticks (double min, double max, double distance, int log_scale, TICKS * ticks)
{
    double step ;	/* Put numbered ticks at multiples of this */
    double range = max - min ;
    int k ;
    double value ;	/* Temporary */
    
    if (log_scale == 1)
        return calculate_log_ticks (min, max, distance, ticks) ;
    
    /* Linear version */
    
    /* Choose a step between successive axis labels so that one digit
     ** changes by 1, 2 or 5 amd that gives us at least the number of
     ** divisions (and numberic labels) that we would like to have.
     **
     ** We do this by starting "step" at the lowest power of ten <= max,
     ** which can give us at most 9 divisions (e.g. from 0 to 9999, step 1000)
     ** Then try 5*this, 2*this and 1*this.
     */
    step = pow (10.0, floor (log10 (max))) ;
    do
    {	if (range / (step * 5) >= TARGET_DIVISIONS)
    {	step *= 5 ;
        break ;
    } ;
        if (range / (step * 2) >= TARGET_DIVISIONS)
        {	step *= 2 ;
            break ;
        } ;
        if (range / step >= TARGET_DIVISIONS)
            break ;
        step /= 10 ;
    } while (1) ;	/* This is an odd loop! */
    
    /* Ensure that the least significant digit that changes gets printed, */
    ticks->decimal_places_to_print = lrint (-floor (log10 (step))) ;
    if (ticks->decimal_places_to_print < 0)
        ticks->decimal_places_to_print = 0 ;
    
    /* Now go from the first multiple of step that's >= min to
     * the last one that's <= max. */
    k = 0 ;
    value = ceil (min / step) * step ;
    
#define add_tick(val, just_a_tick) do \
{	if (val >= min - DELTA && val < max + DELTA) \
{	ticks->value [k] = just_a_tick ? NO_NUMBER : val ; \
ticks->distance [k] = distance * \
(log_scale == 2 \
? /*log*/ (log (val) - log (min)) / (log (max) - log (min)) \
: /*lin*/ (val - min) / range) ; \
k++ ; \
} ; \
} while (0)
    
    /* Add the half-way tick before the first number if it's in range */
    add_tick (value - step / 2, true) ;
    
    while (value <= max + DELTA)
    { 	/* Add a tick next to each printed number */
        add_tick (value, false) ;
        
        /* and at the half-way tick after the number if it's in range */
        add_tick (value + step / 2, true) ;
        
        value += step ;
    } ;
    
    return k ;
} /* calculate_ticks */

/* Number/tick placer for logarithmic scales.
 **
 ** Some say we should number 1, 10, 100, 1000, 1000 and place ticks at
 ** 2,3,4,5,6,7,8,9, 20,30,40,50,60,70,80,90, 200,300,400,500,600,700,800,900
 ** Others suggest numbering 1,2,5, 10,20,50, 100,200,500.
 **
 ** Ticking 1-9 is visually distinctive and emphasizes that we are using
 ** a log scale, as well as mimicking log graph paper.
 ** Numbering the powers of ten and, if that doesn't give enough labels,
 ** numbering also the 2 and 5 multiples might work.
 **
 ** Apart from our [number] and tick styles:
 ** [1] 2 5 [10] 20 50 [100]  and
 ** [1] [2] 3 4 [5] 6 7 8 9 [10]
 ** the following are also seen in use:
 ** [1] [2] 3 4 [5] 6 7 [8] 9 [10]  and
 ** [1] [2] [3] [4] [5] [6] 7 [8] 9 [10]
 ** in https://www.lhup.edu/~dsimanek/scenario/errorman/graphs2.htm
 **
 ** This works fine for wide ranges, not so well for narrow ranges like
 ** 5000-6000, so for ranges less than a decade we apply the above
 ** linear numbering style 0.2 0.4 0.6 0.8 or whatever, but calulating
 ** the positions of the legends logarithmically.
 **
 ** Alternatives could be:
 ** - by powers or two from some starting frequency
 **   defaulting to the Nyquist frequency (22050, 11025, 5512.5 ...) or from some
 **   musical pitch (220, 440, 880, 1760)
 ** - with a musical note scale  C0 ' D0 ' E0 F0 ' G0 ' A0 ' B0 C1
 ** - with manuscript staff lines, piano note or guitar string overlay.
 */

/* Helper functions: add ticks and labels at start_value and all powers of ten
 ** times it that are in the min-max range.
 ** This is used to plonk ticks at 1, 10, 100, 1000 then at 2, 20, 200, 2000
 ** then at 5, 50, 500, 5000 and so on.
 */
static int
add_log_ticks (double min, double max, double distance, TICKS * ticks,
               int k, double start_value, bool include_number)
{	double value ;
    
    for (value = start_value ; value <= max + DELTA ; value *= 10.0)
    {	if (value < min - DELTA) continue ;
        ticks->value [k] = include_number ? value : NO_NUMBER ;
        ticks->distance [k] = distance * (log (value) - log (min)) / (log (max) - log (min)) ;
        k++ ;
    } ;
    return k ;
} /* add_log_ticks */

static int
calculate_log_ticks (double min, double max, double distance, TICKS * ticks)
{	int k = 0 ;	/* Number of ticks we have placed in "ticks" array */
    double underpinning ; 	/* Largest power of ten that is <= min */
    
    /* If the interval is less than a decade, just apply the same
     ** numbering-choosing scheme as used with linear axis, with the
     ** ticks positioned logarithmically.
     */
    if (max / min < 10.0)
        return calculate_ticks (min, max, distance, 2, ticks) ;
    
    /* If the range is greater than 1 to 1000000, it will generate more than
     ** 19 ticks.  Better to fail explicitly than to overflow.
     */
    if (max / min > 1000000)
    {	printf ("Error: Frequency range is too great for logarithmic scale.\n") ;
        exit (1) ;
    } ;
    
    /* First hack: label the powers of ten. */
    
    /* Find largest power of ten that is <= minimum value */
    underpinning = pow (10.0, floor (log10 (min))) ;
    
    /* Go powering up by 10 from there, numbering as we go. */
    k = add_log_ticks (min, max, distance, ticks, k, underpinning, true) ;
    
    /* Do we have enough numbers? If so, add numberless ticks at 2 and 5 */
    if (k >= TARGET_DIVISIONS + 1) /* Number of labels is n.of divisions + 1 */
    {
        k = add_log_ticks (min, max, distance, ticks, k, underpinning * 2.0, false) ;
        k = add_log_ticks (min, max, distance, ticks, k, underpinning * 5.0, false) ;
    }
    else
    {	int i ;
        /* Not enough numbers: add numbered ticks at 2 and 5 and
         * unnumbered ticks at all the rest */
        for (i = 2 ; i <= 9 ; i++)
            k = add_log_ticks (min, max, distance, ticks, k,
                               underpinning * (1.0 * i), i == 2 || i == 5) ;
    } ;
    
    /* Greatest possible number of ticks calculation:
     ** The worst case is when the else clause adds 8 ticks with the maximal
     ** number of divisions, which is when k == TARGET_DIVISIONS, 3,
     ** for example 100, 1000, 10000.
     ** The else clause adds another 8 ticks inside each division as well as
     ** up to 8 ticks after the last number (from 20000 to 90000)
     ** and 8 before to the first (from 20 to 90 in the example).
     ** Maximum possible ticks is 3+8+8+8+8=35
     */
    
    return k ;
} /* calculate_log_ticks */

/* Helper function:
 ** Map the index for an output pixel in a column to an index into the
 ** FFT result representing the same frequency.
 ** magindex is from 0 to maglen-1, representing min_freq to max_freq Hz.
 ** Return values from are from 0 to speclen representing frequencies from
 ** 0 to the Nyquist frequency.
 ** The result is a floating point number as it may fall between elements,
 ** allowing the caller to interpolate onto the input array.
 */
static double
magindex_to_specindex (int speclen, int maglen, int magindex, double min_freq, double max_freq, int samplerate, bool log_freq)
{
    double freq ; /* The frequency that this output value represents */
    
    if (!log_freq)
        freq = min_freq + (max_freq - min_freq) * magindex / (maglen - 1) ;
    else
        freq = min_freq * pow (max_freq / min_freq, (double) magindex / (maglen - 1)) ;
    
    return (freq * speclen / (samplerate / 2)) ;
}

double magSpec(const std::vector<double>& freqDomain, size_t index)
{
    if (index == 0 || index == freqDomain.size()-1)
    {
        return freqDomain.at(index);
    }
    auto const re = freqDomain.at(index);
    auto const im = freqDomain.at(freqDomain.size() - index);
    return sqrt(re * re + im * im);
}

/* Map values from the spectrogram onto an array of magnitudes, the values
 ** for display. Reads spec[0..speclen], writes mag[0..maglen-1].
 */
std::vector<float> interp_spec (const size_t maglen, const std::vector<double>& freqDomain, const double min_freq, const double max_freq, int samplerate)
{
    /* Map each output coordinate to where it depends on in the input array.
     ** If there are more input values than output values, we need to average
     ** a range of inputs.
     ** If there are more output values than input values we do linear
     ** interpolation between the two inputs values that a reverse-mapped
     ** output value's coordinate falls between.
     **
     ** spec points to an array with elements [0..speclen] inclusive
     ** representing frequencies from 0 to samplerate/2 Hz. Map these to the
     ** scale values min_freq to max_freq so that the bottom and top pixels
     ** in the output represent the energy in the sound at min_ and max_freq Hz.
     */
    
    auto mag = std::vector<float>(maglen);
    auto const magSpecLen = (freqDomain.size()/2)+1;
    for (size_t k = 0 ; k < maglen; k++)
    {
        /* Average the pixels in the range it comes from */
        auto current = magindex_to_specindex(magSpecLen, maglen, k, min_freq, max_freq, samplerate, 0.0);
        auto const next = magindex_to_specindex (magSpecLen, maglen, k+1, min_freq, max_freq, samplerate, 0.0);
        
        /* Range check: can happen if --max-freq > samplerate / 2 */
        if (current > magSpecLen)
        {
            mag.at(k) = 0.0;
            return mag;
        }
        
        auto const delta = [](const double value){
            return value - std::floor(value);
        };
        
        if (current + 1.0 < next) {
            /* The output indices are more sparse than the input indices
             ** so average the range of input indices that map to this output,
             ** making sure not to exceed the input array (0..speclen inclusive)
             */
            /* Take a proportional part of the first sample */
            auto count = 1.0 - delta(current);
            auto sum = magSpec(freqDomain, static_cast<size_t>(current)) * count ;
            
            while ((current += 1.0) < next && static_cast<size_t>(current) <= magSpecLen)
            {
                sum += magSpec(freqDomain, static_cast<size_t>(current)) ;
                count += 1.0 ;
            }
            /* and part of the last one */
            if (static_cast<size_t>(next) <= magSpecLen)
            {
                sum += magSpec(freqDomain, static_cast<size_t>(next)) * delta(next) ;
                count += delta(next) ;
            }
            
            mag [k] = sum / count ;
        } else {
            /* The output indices are more densely packed than the input indices
             ** so interpolate between input values to generate more output values.
             */
            /* Take a weighted average of the nearest values */
            auto const deltaCurrent = delta(current);
            mag[k] = magSpec(freqDomain, static_cast<size_t>(current))     * (1.0 - deltaCurrent)
                   + magSpec(freqDomain, static_cast<size_t>(current)+1)   * (deltaCurrent);
        }
    }
    return mag;
}

/* Pick the best FFT length good for FFTW?
 **
 ** We use fftw_plan_r2r_1d() for which the documantation
 ** http://fftw.org/fftw3_doc/Real_002dto_002dReal-Transforms.html says:
 **
 ** "FFTW is generally best at handling sizes of the form
 ** 2^a 3^b 5^c 7^d 11^e 13^f
 ** where e+f is either 0 or 1, and the other exponents are arbitrary."
 */

/* Helper function: does N have only 2, 3, 5 and 7 as its factors? */
static bool
is_2357 (int n)
{
    /* Just eliminate all factors os 2, 3, 5 and 7 and see if 1 remains */
    while (n % 2 == 0) n /= 2 ;
    while (n % 3 == 0) n /= 3 ;
    while (n % 5 == 0) n /= 5 ;
    while (n % 7 == 0) n /= 7 ;
    return (n == 1) ;
}

/* Helper function: is N a "fast" value for the FFT size? */
static bool
is_good_speclen (int n)
{
    /* It wants n, 11*n, 13*n but not (11*13*n)
     ** where n only has as factors 2, 3, 5 and 7
     */
    if (n % (11 * 13) == 0) return 0 ; /* No good */
    
    return is_2357 (n)	|| ((n % 11 == 0) && is_2357 (n / 11))
    || ((n % 13 == 0) && is_2357 (n / 13)) ;
}

struct Spectrogram {
    const long long w, h;
};

namespace time_ {
    
    using Duration = double;
    using Frequency = double;
    
    Duration fromSeconds(double seconds) {
        return seconds;
    }

    Duration parseDuration(const char* value) {
        return fromSeconds(atof(value));
    }

    template <typename DurationType>
    DurationType fromRatio(long long num, long long denom) {
        if (denom == 0) {
            throw std::runtime_error{"division by zero (num=" + std::to_string(num) + ")"};
        }
        return static_cast<Duration>(num) / static_cast<Duration>(denom);
    }

}

using Duration = time_::Duration;
using Frequency = time_::Frequency;

int main
(
 int ac,
 char *av[]
 )
{
    if (ac != 7)
    {
        printf("usage: %s [sound file] [win secs] [step secs] [export image height] [cfg file] [weights file]\n", av[0]);
        exit(1);
    }
    
    auto infile = SndfileHandle(av[1]);
    if (!infile)
    {
        printf("failed to open input sound file: %s\n", sf_strerror(NULL));
        exit(1);
    }
    
    constexpr auto SPEC_FLOOR_DB = -180.0;
    constexpr auto MIN_FREQ = 0.0;
    constexpr auto PIXEL_WIDTH_PER_SECOND = 100;
    constexpr auto MAG_TO_NORMALIZE = 100.0;

    const double LINEAR_SPEC_FLOOR = pow(10.0, SPEC_FLOOR_DB / 20.0);

    struct Program {
      
        struct Lengths {
            Duration total, window, step;
        };
        
        static std::optional<Program> create(const Frequency maxFrequency, const Lengths lengths) {
            if (lengths.window > lengths.total) {
                return std::nullopt;
            }
            return Program{ maxFrequency, lengths };
        }
        
        const Frequency maxFrequency;
        const Lengths lengths;
        
        
        int64_t steps() const
        {
            return 1 + (lengths.total-lengths.window)/lengths.step;
        }
        
    private:
        
    };
    
    auto const maxFreq_{time_::fromRatio<Frequency>(infile.samplerate(), 2)};
    auto const totalLength_{time_::fromRatio<Duration>(infile.frames(), infile.samplerate())};
    auto const windowLength_{time_::parseDuration(av[2])};
    auto const stepLength_{time_::parseDuration(av[3])};
    
    auto const program{Program::create(maxFreq_, {totalLength_, windowLength_, stepLength_}).value()};
    
    Spectrogram spectrogram{static_cast<long long>(PIXEL_WIDTH_PER_SECOND * program.lengths.window), atoll(av[4])};
    
    auto speclen = spectrogram.h * (infile.samplerate() / 20 / spectrogram.h + 1);
    for (auto i = 0ll; ; ++i)
    {
        if (is_good_speclen(speclen + i))
        {
            speclen += i;
            break;
        }
        if (speclen - i >= spectrogram.h && is_good_speclen(speclen - i))
        {
            speclen -= i;
            break;
        }
    }
    
    using std::vector;
    
    auto timeDomain = vector<double>(2*speclen+1);
    vector<double> freqDomain(2*speclen);
    
    auto spec = Spectrum::create(speclen, timeDomain.data(), freqDomain.data()).value();
    
    auto magSpecMatrix = vector<vector<float>>(spectrogram.w, vector<float>(spectrogram.h));
    
    cv::Mat im(spectrogram.h, spectrogram.w, CV_8UC3);
    unsigned char colour[3] = {0, 0, 0};
    
    
    
    auto net = Network::init(av[5], av[6], spectrogram.w, spectrogram.h, 3).value();
    
    for (auto i = 0ll; i < program.steps(); ++i)
    {
        for (auto j = 0ll; j < spectrogram.w; ++j)
        {
            
            
            std::fill(begin(timeDomain), end(timeDomain), 0.0);
            auto data = timeDomain.data();
            int datalen = timeDomain.size();
                        
            sf_count_t start = ((j + i * program.lengths.step * spectrogram.w / program.lengths.window) * infile.samplerate() * program.lengths.window) / spectrogram.w - speclen;
            if (start >= 0)
            {
                infile.seek(start, SEEK_SET);
            }
            else
            {
                start = -start;
                infile.seek(0, SEEK_SET);
                data += start;
                datalen -= start;
            }
            if (infile.channels() == 1)
            {
                infile.read(data, datalen);
            }
            else
            {
                // mix channels
                int ch = 0;
                int frames_read = 0;
                sf_count_t dataout = 0;
                static double multi_data[2048];
                while (dataout < datalen)
                {
                    int this_read = MIN(ARRAY_LEN(multi_data) / infile.channels(), datalen - dataout);
                    frames_read = infile.readf(multi_data, this_read);
                    if (frames_read == 0)
                    {
                        break;
                    }
                    for (auto k = 0; k < frames_read; ++k)
                    {
                        double mix = 0.0;
                        for (ch = 0; ch < infile.channels(); ++ch)
                        {
                            mix += multi_data[k * infile.channels() + ch];
                        }
                        data[dataout + k] = mix / infile.channels();
                    }
                    dataout += frames_read;
                }
            }
            spec.applyWindow(timeDomain.data(), timeDomain.size());
            spec.executeFft();
            magSpecMatrix[j] = interp_spec(spectrogram.h, freqDomain, MIN_FREQ, program.maxFrequency, infile.samplerate());
        }
        
        // draw spectrogram
        for (auto j = 0ll; j < spectrogram.w; ++j)
        {
            for (auto k = 0ll; k < spectrogram.h; ++k)
            {
                magSpecMatrix[j][k] /= MAG_TO_NORMALIZE;
                magSpecMatrix[j][k] = (magSpecMatrix[j][k] < LINEAR_SPEC_FLOOR) ? SPEC_FLOOR_DB : 20.0 * log10(magSpecMatrix[j][k]);
                get_colour_map_value(magSpecMatrix[j][k], SPEC_FLOOR_DB, colour);
                im.data[((spectrogram.h - 1 - k) * im.cols + j) * 3] = colour[2];
                im.data[((spectrogram.h - 1 - k) * im.cols + j) * 3 + 1] = colour[1];
                im.data[((spectrogram.h - 1 - k) * im.cols + j) * 3 + 2] = colour[0];
            }
        }
        

        auto const netout = net.run(im.data);
        if (netout[0] > 0.9f)
        {
            printf("woof woof!\n");
            cv::rectangle(im, cv::Rect(20, 10, spectrogram.w - 40, spectrogram.h - 20), cv::Scalar(255, 0, 0), 15, 16);
            cv::putText(im, "dog bark", cv::Point(spectrogram.w/2-40, spectrogram.h/2), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 0, 0), 2, 16);
        }
        cv::imshow("demo", im);
        
        unsigned char key = cv::waitKey(10);
        if (key == 27)
        {
            break;
        }
    }
    
    return 0 ;
}

