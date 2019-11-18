/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once
#include "c74_min.h"

using namespace c74::min;

class ddsp_tilde : public object<ddsp_tilde>, public sample_operator<2, 1> {
public:
	MIN_DESCRIPTION	{ "Implements the Differentiable Digital Signal Processing into"
 									 "MAX / MSP with libtorch."};
	MIN_TAGS		{ "synthesis" };
	MIN_AUTHOR		{ "Antoine CAILLON" };
	// MIN_RELATED		{ "min.edge~, min.edgelow~, edge~, snapshot~, ==~" };

	inlet<>  f0_input	{ this, "Fudamental frequency (f0)" };
	inlet<>  lo_input	{ this, "Loudness (lo)" };
	outlet<> output	{ this, "Output of DDSP" };

	sample operator()(sample f0, sample lo) {
		return f0;
	}

private:

};

MIN_EXTERNAL(ddsp_tilde);
