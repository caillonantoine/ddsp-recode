#include "ddsp_core.hpp"
#include "c74_min.h"
#include <math.h>
#include <algorithm>

#define PI             3.14159265359
#define PARTIAL_NUMBER 100
#define FILTER_SIZE    81
#define SAMPLERATE     16000

using namespace c74::min;

class ddsp_tilde : public object<ddsp_tilde>, public sample_operator<2, 1> {
public:
	MIN_DESCRIPTION	{ "Implements the Differentiable Digital Signal Processing into"
 									 "MAX / MSP with libtorch."};
	MIN_TAGS		{ "synthesis" };
	MIN_AUTHOR		{ "Antoine CAILLON" };

	inlet<>  f0_input	{ this, "Fudamental frequency (f0)" };
	inlet<>  lo_input	{ this, "Loudness (lo)" };
	outlet<> output	{ this, "Output of DDSP", "signal" };

	// ddsp_tilde(){
		// model = new DDSPCore();
	// }

	sample operator()(sample f0, sample lo) {

		float output(0);

		// if (head++ % 160 == 0) {
			// model->getNextOutput(float(f0), float(lo), parameters);
			// head = 0;
		// }

		// SYNTH
		for (int i(0); i<30; i++) {
			output += cos ( i * i_phase );
		}

		// UPDATE INSTANTANEOUS PHASE
		i_phase += 2 * PI * float(f0) / SAMPLERATE;
		while (i_phase > 2 * PI) { i_phase -= 2 * PI;}


		return output;
	}

private:
	float i_phase;
	int head;
	// float parameters[PARTIAL_NUMBER + FILTER_SIZE + 1];
	// DDSPCore *model;

};

MIN_EXTERNAL(ddsp_tilde);
