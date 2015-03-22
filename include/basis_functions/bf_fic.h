// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_FIC_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_FIC_H_

#include "IBasisFunction.h"

namespace libgp {
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

	void gradDiagWrapped(SampleSet * sampleset, const Eigen::VectorXd & diagK,
			size_t parameter, Eigen::VectorXd & gradient);

	bool gradDiagWrappedIsNull(size_t parameter);

	virtual void gradBasisFunction(SampleSet * sampleSet, const Eigen::MatrixXd &Phi,
			size_t p, Eigen::MatrixXd &Grad);

	bool gradBasisFunctionIsNull(size_t p);

	virtual void gradiSigma(size_t p, Eigen::MatrixXd & diSigmadp);

	bool gradiSigmaIsNull(size_t p);

	bool sigmaIsDiagonal() {
		return false;
	}
	;

	virtual std::string to_string();

	std::string pretty_print_parameters();

	/**
	 * Inducing point matrix.
	 *
	 * TODO: This is a hack to make the optimized version work.
	 */
	Eigen::MatrixXd U;
protected:
	void log_hyper_updated(const Eigen::VectorXd &p);

	virtual bool real_init();

	size_t get_param_dim_without_noise(size_t input_dim,
			size_t num_basis_functions);

	Eigen::MatrixXd Sigma;

	Eigen::MatrixXd iSigma;

	Eigen::MatrixXd choliSigma;

	double logDetSigma;

	/**
	 * Vector containing the parameters of the covariance function.
	 */
	Eigen::VectorXd cov_params;

	double half_log_det_sigma;

	/**
	 * Inducing input noise.
	 */
	double snu2;

	/**
	 * Contains cov_params.size().
	 */
	size_t cov_params_size;

	/**
	 * Temporary vector of the same size as cov_params.
	 */
	Eigen::VectorXd temp_cov_params_size;

	/**
	 * Temporary vector of size input_dim.
	 */
	Eigen::VectorXd temp_input_dim;
};
}

#endif /* INCLUDE_BASIS_FUNCTIONS_BF_FIC_H_ */
