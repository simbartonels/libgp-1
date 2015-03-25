// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__SRC_GP_MS_OPTIMIZED_H_
#define SOURCE_DIRECTORY__SRC_GP_MS_OPTIMIZED_H_

#include "gp_fic.h"

namespace libgp{

/**
 * Optimizes the gradient computation. TODO: REFACTOR! The whole architecture of this optimization is bad and should be
 * integrated into gp_fic instead!
 */
class OptMultiScaleGaussianProcess: public FICGaussianProcess {
public:
	/** Create and instance of GaussianProcess with given input dimensionality
	 *  and covariance function. */
	OptMultiScaleGaussianProcess(size_t input_dim, std::string covf_def,
			size_t num_basisf, std::string basisf_def);

	/** Create and instance of GaussianProcess from file. */
	virtual ~OptMultiScaleGaussianProcess(){};

protected:

	double grad_basis_function(size_t i, bool gradBasisFunctionIsNull, bool gradiSigmaIsNull);

	double grad_isigma(size_t i, bool gradiSigmaIsNull);

private:
	/**
	 * A vector that represents the gradient dKuui.
	 */
	Eigen::VectorXd dkuui;

	Eigen::VectorXd temp_input_dim;

	size_t m;

	size_t d;

	bool optimize;
};
}
#endif
