// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__SRC_GP_APPR_H_
#define SOURCE_DIRECTORY__SRC_GP_APPR_H_

#include <cmath>
#include <Eigen/Dense>

#include "abstract_gp.h"
#include "cov.h"
#include "sampleset.h"
#include "basis_functions/IBasisFunction.h"

namespace libgp {

/**
 * Approximate Gaussian process regression using FIC.
 * See "Approximation Methods for Gaussian Processes" by Quinonero-Candela,
 * Rasmussen and Williams from 2007. The implementation follows the framework written
 * by Chalupka. See "A Framework for Evaluating Approximation Methods for Gaussian Process
 * Regression" by Chalupka, Williams and Murray from 2013.
 *
 *  @author Manuel Blum, Simon Bartels
 */
class FICGaussianProcess: public AbstractGaussianProcess {
public:

	/** Create and instance of GaussianProcess with given input dimensionality
	 *  and covariance function. */
	FICGaussianProcess(size_t input_dim, std::string covf_def,
			size_t num_basisf, std::string basisf_def); // : AbstractGaussianProcess(input_dim, covf_def){};
	/** Create and instance of GaussianProcess from file. */
//    FICGaussianProcess (const char * filename) : AbstractGaussianProcess(filename) {};
	virtual ~FICGaussianProcess();

protected:
	double var_impl(const Eigen::VectorXd &x_star);

	void grad_var_impl(const Eigen::VectorXd & x, Eigen::VectorXd & grad);

	double log_likelihood_impl();

	virtual Eigen::VectorXd log_likelihood_gradient_impl();

	/** Update test input and cache kernel vector. */
	void update_k_star(const Eigen::VectorXd &x_star);

	void update_alpha();

	/** Compute covariance matrix and perform cholesky decomposition. */
	void computeCholesky();

	void updateCholesky(const double x[], double y);

	/**
	 * Computes the basis function part of the log-likelihood gradient.
	 */
	virtual double grad_basis_function(size_t i, bool gradBasisFunctionIsNull, bool gradiSigmaIsNull);

	/**
	 * Computes the weight prior part of the log-likelihood gradient.
	 */
	virtual double grad_isigma(size_t i, bool gradiSigmaIsNull);

	Eigen::MatrixXd B;
	Eigen::MatrixXd R;
	Eigen::MatrixXd dKui;
	Eigen::MatrixXd dKuui;
	Eigen::VectorXd al;
	/**
	 * Corresponds to Ku in infFITC.
	 */
	Eigen::MatrixXd Phi;
	Eigen::VectorXd w;
	Eigen::VectorXd v;

	/**
	 * Convenience pointer that just points to the covariance function cf in the super class.
	 */
	IBasisFunction * bf;

	/**
	 * Number of basis functions.
	 */
	size_t M;
	/**
	 * Initializes all the vectors and matrices used in the for loop.
	 */
	void log_likelihood_gradient_precomputations();

	/**
	 * Corresponds to diagK in infFITC.
	 */
	Eigen::VectorXd k;

	Eigen::MatrixXd Lu;
	Eigen::VectorXd dg;
	Eigen::VectorXd alSqrd;
	Eigen::VectorXd isqrtgamma;
	Eigen::MatrixXd V;
	Eigen::MatrixXd W;
	Eigen::MatrixXd Wdg;
	Eigen::MatrixXd BWdg;
	Eigen::VectorXd WdgSum;

	Eigen::VectorXd ddiagK;

	Eigen::VectorXd r;
	Eigen::VectorXd beta;

	Eigen::VectorXd tempM;

	/**
	 * Temporary matrix to save the product of Luu and Lu.
	 */
	Eigen::MatrixXd LuuLu;

};
}

#endif /* SOURCE_DIRECTORY__SRC_GP_APPR_H_ */
