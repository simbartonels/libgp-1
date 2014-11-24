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

	/**
	 * Constant used by functions of the form grad***info(). States that there is no additional
	 * information about the matrix.
	 */
	static const int IBF_MATRIX_INFO_NONE = 0;

	/**
	 * Constant used by functions of the form grad***info(). States that the matrix is diagonal.
	 */
	static const int IBF_MATRIX_INFO_DIAG = 1;

	/**
	 * Constant used by functions of the form grad***info(). States that the matrix is all zero.
	 */
	static const int IBF_MATRIX_INFO_NULL = 2;

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
	 * @param x the input to phi
	 * @param phi phi(x)
	 * @param p the number of the parameter
	 * @param grad where to put the result
	 */
	virtual void gradBasisFunction(const Eigen::VectorXd &x,
			const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad) = 0;

	/**
	 * Returns additional information about properties of the gradient of the basis function.
	 * Note: IBF_MATRIX_INFO_DIAG is ignored.
	 * @param p the index of the parameter
	 * @returns one of the constants IBF_MATRIX_INFO_*
	 */
	virtual int gradBasisFunctionInfo(size_t p) {
		return IBF_MATRIX_INFO_NONE;
	}

	/**
	 * Returns the inverse correlation matrix of the Gaussian weight prior
	 * for the basis functions.
	 */
	virtual Eigen::MatrixXd getInverseWeightPrior() = 0;

	/**
	 * Returns the Cholesky of the inverse correlation matrix of the Gaussian weight prior
	 * for the basis functions.
	 */
	virtual Eigen::MatrixXd getCholeskyOfInverseWeightPrior() = 0;

	/**
	 * Returns the weight prior.
	 */
	virtual Eigen::MatrixXd getWeightPrior() = 0;

	/**
	 * Returns log(|Sigma|)/2. Note: MUST return HALF of the log determinant.
	 */
	virtual double getLogDeterminantOfWeightPrior() = 0;

	/**
	 * Computes the derivative of the weight prior with respect to parameter number param.
	 * @param p the number of the parameter
	 * @param diSigmadp where to put the derivative
	 */
	virtual void gradInverseWeightPrior(size_t p,
			Eigen::MatrixXd & diSigmadp) = 0;

	/**
	 * Gives some extra information about the gradient of the inverse weight prior, e.g.
	 * if it is diagonal, all zero or nothing special.
	 * @param p the index of the parameter
	 * @returns one of the constants IBF_MATRIX_INFO_*
	 */
	virtual int gradInverseWeightPriorInfo(size_t p) {
		return IBF_MATRIX_INFO_NONE;
	}
	;

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
	 * Returns the approximated kernel value.
	 */
	double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) {
		Eigen::VectorXd phix = computeBasisFunctionVector(x1);
		Eigen::VectorXd phiz = computeBasisFunctionVector(x2);
		Eigen::MatrixXd L = getCholeskyOfInverseWeightPrior();
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

protected:
	/**
	 * Performs the actual initialization.
	 */
	virtual bool real_init() = 0;

	size_t M;

	CovarianceFunction * cov;
};
}

#endif /* SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_ */
