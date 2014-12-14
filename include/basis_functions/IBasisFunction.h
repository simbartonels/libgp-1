// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_
#define SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_

#include "cov.h"

namespace libgp {
class IBasisFunction: public CovarianceFunction {
	/**
	 * Interface for basis functions for degenerate kernels.
	 *
	 * ATTENTION: The last hyper-parameter MUST be noise!
	 */
public:
	//constructor conversions
	//IBasisFunction(const CovarianceFunction & cf){};

	bool init(int input_dim) {
		//TODO: give a signal that this is not the way to initialize basis functions
		std::cout
				<< "IBasisFunction: Wrong initialization method for basis functions!"
				<< std::endl;
		return false;
	}

	/** Initialization method for atomic basis functions.
	 *  @param M the number of basis functions
	 *  @param wrappedCovFunc the wrapped covariance function
	 *  @return true if initialization was successful.
	 */
	bool init(size_t M, CovarianceFunction * wrappedCovFunc) {
		input_dim = wrappedCovFunc->get_input_dim();

		this->M = M;
		cov = wrappedCovFunc;
		return real_init();
	}
	;

	/**
	 * Computes the values of all basis functions for a given vector.
	 * The underlying input distribution and the number of basis
	 * functions should be determined during construction.
	 * @param x input vector
	 * @return the vector of basis function values
	 */
	virtual Eigen::VectorXd computeBasisFunctionVector(
			const Eigen::VectorXd &x) = 0;

	/**
	 * Computes the derivative of a basis function vector phi(x) with respect to parameter i.
	 * Implementing classes MAY assume for the first call that grad is 0 initialized.
	 * @param x the input to phi
	 * @param phi phi(x)
	 * @param p the number of the parameter
	 * @param grad where to put the result
	 */
	virtual void gradBasisFunction(const Eigen::VectorXd &x,
			const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad) = 0;

	/**
	 * Returns additional information about properties of the gradient of the basis function.
	 * @param p the index of the parameter
	 * @returns True if the gradient is the zero matrix.
	 */
	virtual bool gradBasisFunctionIsNull(size_t p) = 0;

	/**
	 * Returns the covariance matrix of the weight prior. I.e.
	 * the matrix Sigma for which k(x,z)=phi(x) Sigma phi(z).
	 */
	virtual Eigen::MatrixXd getSigma() = 0;


	/**
	 * Returns true if the matrix Sigma is a diagonal matrix.
	 */
	virtual bool sigmaIsDiagonal() = 0;

	/**
	 * Returns log(|Sigma|)/2. Note: MUST return HALF of the log determinant.
	 * TODO: implement and allow override
	 */
	virtual double getLogDeterminantOfSigma() = 0;

	/**
	 * Returns the inverse of Sigma.
	 * TODO: implement and allow override
	 */
	virtual Eigen::MatrixXd getInverseOfSigma() = 0;

	/**
	 * Returns the Cholesky of Sigma.
	 * TODO: implement and allow override
	 */
	virtual Eigen::MatrixXd getCholeskyOfInvertedSigma() = 0;

	/**
	 * Computes the derivative of Sigma^-1 with respect to parameter number param.
	 * Implementing classes MAY assume for the first call that diSigmadp is 0 initialized.
	 *
	 * @param p the number of the parameter
	 * @param diSigmadp where to put the derivative
	 */
	virtual void gradiSigma(size_t p,
			Eigen::MatrixXd & diSigmadp) = 0;

	/**
	 * Gives some extra information about the gradient Sigma^-1.
	 * @param p the index of the parameter
	 * @returns True if the gradient is the null matrix.
	 */
	virtual bool gradiSigmaIsNull(size_t p) = 0;

	/**
	 * Returns the actual number of basis functions in use.
	 */
	size_t getNumberOfBasisFunctions() {
		return M;
	}

	/**
	 * Returns what the original kernel would return.
	 */
	virtual double getWrappedKernelValue(const Eigen::VectorXd &x1,
			const Eigen::VectorXd &x2) {
		return cov->get(x1, x2);
	}

	/**
	 * Returns the approximated kernel value. I.e. k(x,z)=phi(x) Sigma phi(z).
	 */
	double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) {
		//TODO: refactor
		Eigen::VectorXd phix = computeBasisFunctionVector(x1);
		Eigen::VectorXd phiz = computeBasisFunctionVector(x2);
		Eigen::MatrixXd L = getCholeskyOfInvertedSigma();
		Eigen::VectorXd r;
		r = L * phix;
		r = r.transpose() * L * phiz;
		return r(0, 0);
	}

	/**
	 * Returns the noise on a log scale.
	 * ATTENTION: The noise MUST be the last parameter!
	 * This is necessary for hyper-parameter optimization for degenerate kernels.
	 * Unfortunately, there seems to be no easy way to hide this parameter.
	 */
	double getLogNoise(){
		return loghyper(get_param_dim() - 1);
	}

	void set_loghyper(const Eigen::VectorXd &p){
		CovarianceFunction::set_loghyper(p);
		//TODO: think about noise!
		log_hyper_updated(p);
	};

protected:
	/**
	 * Performs the actual initialization.
	 */
	virtual bool real_init() = 0;

	/**
	 * Notifies the basis function that new hyper-parameters arrived.
	 */
	virtual void log_hyper_updated(const Eigen::VectorXd &p) = 0;

	size_t M;

	CovarianceFunction * cov;
};
}

#endif /* SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_ */
