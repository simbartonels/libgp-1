// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_FIC_FIXED_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_FIC_FIXED_H_

#include "bf_fic.h"

namespace libgp {

/**
 * Class that implements the Fully Independent Conditional approximation but fixes the inducing
 * inputs in the beginning.
 */
class FICfixed: public FIC {
public:
	virtual ~FICfixed();

	void gradBasisFunction(SampleSet * sampleSet, const Eigen::MatrixXd &Phi,
			size_t p, Eigen::MatrixXd &Grad);

	void gradiSigma(size_t p, Eigen::MatrixXd & diSigmadp);

	std::string to_string();

	/**
	 * Sets the location of the inducing points.
	 */
	void setExtraParameters(const Eigen::MatrixXd & U);
protected:
	void log_hyper_updated(const Eigen::VectorXd &p);

	size_t get_param_dim_without_noise(size_t input_dim,
			size_t num_basis_functions);
private:
	bool U_initialized;
};
}

#endif /* INCLUDE_BASIS_FUNCTIONS_BF_FIC_FIXED_H_ */
