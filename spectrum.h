#ifndef SPECTRUM_H
#define SPECTRUM_H
#endif

#ifdef __cplusplus
extern "C"{
#endif

typedef struct spectrum spectrum;
typedef enum WINDOW_FUNCTION WINDOW_FUNCTION;

spectrum * create_spectrum (int speclen, WINDOW_FUNCTION window_function) ;

void destroy_spectrum (spectrum * spec) ;

double calc_magnitude_spectrum (spectrum * spec) ;

double * spectrum_time_domain(spectrum*);
double * spectrum_mag_spec(spectrum*);

#ifdef __cplusplus
}
#endif

