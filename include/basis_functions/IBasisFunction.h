// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_
#define SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_

#include "cov.h"

namespace libgp {
class IBasisFunction : public CovarianceFunction {
public:

	//constructor conversions
	//IBasisFunction(const CovarianceFunction & cf){};

	bool init(int input_dim){
		//TODO: give a signal that this is not the way to initialize basis functions
		std::cout << "IBasisFunction: Wrong initialization method for basis functions!" << std::endl;
		return false;
	}

	/** Initialization method for atomic basis functions.
     *  @param M the number of basis functions
     *  @param wrappedCovFunc the wrapped covariance function
     *  @return true if initialization was successful.
     */
	bool init(size_t M, CovarianceFunction * wrappedCovFunc)
	{
		input_dim = wrappedCovFunc->get_input_dim();

		this->M = M;
		cov = wrappedCovFunc;
		return real_init();
	};

	/**
	 * Computes the values of all basis functions for a given vector.
	 * The underlying input distribution and the number of basis
	 * functions should be determined during construction.
	 * @param x input vector
	 * @return the vector of basis function values
	 */
	virtual Eigen::VectorXd computeBasisFunctionVector(const Eigen::VectorXd &x) = 0;

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
	 * Returns the actual number of basis functions in use.
	 */
	size_t getNumberOfBasisFunctions(){
		return M;
	}

    /**
     * Returns what the original kernel would return.
     */
    virtual double getWrappedKernelValue(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2){
    	return cov->get(x1, x2);
    }

    /**
     * Returns the approximated kernel value.
     */
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
    {
    	Eigen::VectorXd phix = computeBasisFunctionVector(x1);
    	Eigen::VectorXd phiz = computeBasisFunctionVector(x2);
    	Eigen::MatrixXd L = getCholeskyOfInverseWeightPrior();
    	Eigen::VectorXd r;
    	r = L*phix;
    	r = r.transpose() * L * phiz;
    	return r(0, 0);
    }

//    virtual void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad) = 0;
//
//    virtual void set_loghyper(const Eigen::VectorXd &p);
//
//    virtual void set_loghyper(const double p[]);
//
//    virtual size_t get_param_dim();
//
//    virtual Eigen::VectorXd get_loghyper();
//
//    virtual std::string to_string();


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
