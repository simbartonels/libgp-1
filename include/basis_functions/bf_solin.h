// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_SOLIN_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_SOLIN_H_

#include "IBasisFunction.h"

namespace libgp {
/**
 * Implements "Hilbert Space Methods for Reduced Rank Gaussian Process Regression" by Solin
 * and Särkkä from 2014. This implementation is taylored to the Squared Exponential but therefore
 * a bit more clever as it allows automatic relevance determination without sacrificing O(M^3)
 * hyper-parameter optimization. For a derivation see my thesis.
 *
 * @author Simon Bartels
 */
class Solin: public IBasisFunction {
public:
	virtual ~Solin();

	Eigen::VectorXd computeBasisFunctionVector(const Eigen::VectorXd &x);

	Eigen::MatrixXd getInverseOfSigma();

	Eigen::MatrixXd getCholeskyOfInvertedSigma();

	Eigen::MatrixXd getSigma();

	double getLogDeterminantOfSigma();

	void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
			Eigen::VectorXd &grad);

	void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
			double kernel_value, Eigen::VectorXd &grad);

	void gradBasisFunction(const Eigen::VectorXd &x, const Eigen::VectorXd &phi,
			size_t p, Eigen::VectorXd &grad);

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

private:
	/**
	 * Contains the borders of the input domain. We will assume the input domain is
	 * standardized to length 1.
	 */
	static const double L = 1.2;

	/**
	 * Contains the length scales.
	 */
	Eigen::VectorXd ell;

	/**
	 * Contains the amplitude.
	 */
	double sf2;

	/**
	 * This will be the matrix Gamma from the paper.
	 */
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Sigma;

	Eigen::DiagonalMatrix<double, Eigen::Dynamic> iSigma;

	Eigen::DiagonalMatrix<double, Eigen::Dynamic> choliSigma;

	double logDetSigma;

	/**
	 * The internal number of basis functions, that satisfies pow(input_dim, M_intern)<=M.
	 */
	size_t M_intern;

	/**
	 * Contains the numbers from 1 to M_intern multiplied with pi/2.
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

};
}

#endif /* INCLUDE_BASIS_FUNCTIONS_BF_SOLIN_H_ */
