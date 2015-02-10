// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_SOLIN_NAIVE_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_SOLIN_NAIVE_H_

#include "basis_functions/IBasisFunction.h"

namespace libgp {
/**
 * Implements "Hilbert Space Methods for Reduced Rank Gaussian Process Regression" by Solin
 * and S�rkk� from 2014. This implementation is taylored to the Squared Exponential but therefore
 * a bit more clever as it allows automatic relevance determination without sacrificing O(M^3)
 * hyper-parameter optimization. For a derivation see my thesis.
 *
 * @author Simon Bartels
 */
class SolinNaive: public IBasisFunction {
public:
	SolinNaive();

	virtual ~SolinNaive();

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
		return true;
	}
	;

	std::string to_string();
protected:
	void log_hyper_updated(const Eigen::VectorXd &p);

	virtual bool real_init();

	size_t get_param_dim_without_noise(size_t input_dim,
			size_t num_basis_functions);

private:
	/**
	 * Function computing the spectral density of the ARD squared exponential kernel.
	 *
	 * @param lambdaSquared A vector of size input_dim. Should be lambda^2 (component-wise).
	 * @returns The value of the spectral density.
	 *
	 * TODO: generalize by taking the spectral density from the covariance function where possible.
	 */
	inline double spectralDensity(const Eigen::VectorXd & lambdaSquared);

	/**
	 * Increases the counter 'counter'. 'counter' is supposed to be a vector of size input_dim.
	 * The first dimension is always increased by one modulo M_intern. If after that the value is
	 * zero the next dimension is increased and so on.
	 *
	 * @param counter The counter to be increased.
	 */
	inline void incCounter(Eigen::VectorXi & counter);

	/**
	 * Computes the basis function vector for one dimension.
	 * @param xd The d-th entry of a vector x.
	 * @param phi Where to write the output to.
	 */
	inline void phi1D(const double & xd, Eigen::VectorXd & phi);

	/**
	 * Contains the borders of the input domain. We will assume the input domain is
	 * standardized to length 1.
	 */
	//TODO: make this static const
	const double L;
	const double sqrtL;

	/**
	 * Contains the length scales.
	 */
	Eigen::VectorXd ell;

	/**
	 * Contains the amplitude.
	 */
	double sf2;

	/**
	 * Factor for the spectral density. See log_hyper_updated for the definition.
	 */
	double c;

	/**
	 * (PI/L/2)^2.
	 */
	double piOverLOver2Sqrd;

	/**
	 * This will be the matrix Gamma from the paper.
	 */
	Eigen::MatrixXd Sigma;

	Eigen::MatrixXd iSigma;

	Eigen::MatrixXd choliSigma;

	double logDetSigma;

	/**
	 * The internal number of basis functions, that satisfies pow(input_dim, M_intern)<=M.
	 */
	size_t M_intern;

	/**
	 * Contains the numbers from 1 to M_intern multiplied with pi/L/2.
	 */
	Eigen::VectorXd m;

	/**
	 * Is used in computeBasisFunctions. Holds the values for the 1 dimenensional case.
	 */
	Eigen::VectorXd phi_1D;

	/**
	 * Contains pow(M_intern, input_dim).
	 */
	size_t MToTheD;

	/**
	 * Counter vector.
	 */
	Eigen::VectorXi counter;
};
}

#endif /* INCLUDE_BASIS_FUNCTIONS_BF_SOLIN_NAIVE_H_ */