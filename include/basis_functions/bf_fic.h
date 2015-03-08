// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_FIC_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_FIC_H_

#include "IBasisFunction.h"

namespace libgp {
/**
 * Implements "Hilbert Space Methods for Reduced Rank Gaussian Process Regression" by Solin
 * and S�rkk� from 2014. This implementation is taylored to the Squared Exponential but therefore
 * a bit more clever as it allows automatic relevance determination without sacrificing O(M^3)
 * hyper-parameter optimization. For a derivation see my thesis.
 *
 * @author Simon Bartels
 */
class FIC: public IBasisFunction {
public:
	FIC();

	virtual ~FIC();

	Eigen::VectorXd computeBasisFunctionVector(const Eigen::VectorXd &x);

	const Eigen::MatrixXd & getInverseOfSigma();

	const Eigen::MatrixXd & getCholeskyOfInvertedSigma();

	const Eigen::MatrixXd & getSigma();

	double getLogDeterminantOfSigma();

	void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
			Eigen::VectorXd &grad);

	bool gradDiagWrappedIsNull(size_t parameter);

	void gradBasisFunction(SampleSet * sampleSet, const Eigen::MatrixXd &Phi,
			size_t p, Eigen::MatrixXd &Grad);

	bool gradBasisFunctionIsNull(size_t p);

	void gradiSigma(size_t p, Eigen::MatrixXd & diSigmadp);

	bool gradiSigmaIsNull(size_t p);

	bool sigmaIsDiagonal() {
		return false;
	}
	;

	std::string to_string();

	std::string pretty_print_parameters();
protected:
	void log_hyper_updated(const Eigen::VectorXd &p);

	virtual bool real_init();

	size_t get_param_dim_without_noise(size_t input_dim,
			size_t num_basis_functions);

//	double getLogNoise();

private:
	Eigen::MatrixXd Sigma;

	Eigen::MatrixXd iSigma;

	Eigen::MatrixXd choliSigma;

	double logDetSigma;

	/**
	 * Inducing point matrix.
	 */
	Eigen::MatrixXd U;

	/**
	 * Vector containing the parameters of the covariance function.
	 */
	Eigen::VectorXd cov_params;

	double half_log_det_sigma;
};
}

#endif /* INCLUDE_BASIS_FUNCTIONS_BF_FIC_H_ */
