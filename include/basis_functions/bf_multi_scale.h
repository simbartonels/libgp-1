// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_MULTI_SCALE_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_MULTI_SCALE_H_

#include "IBasisFunction.h"

namespace libgp{
class MultiScale : public IBasisFunction{
public:
		Eigen::VectorXd computeBasisFunctionVector(const Eigen::VectorXd &x);

		Eigen::MatrixXd getInverseWeightPrior();

		Eigen::MatrixXd getCholeskyOfInverseWeightPrior();

		Eigen::MatrixXd getWeightPrior();

		double getLogDeterminantOfWeightPrior();

		/**
		 * Parent is overwritten since this kernel does not exactly wrap the ARDse.
		 */
		double getWrappedKernelValue(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);

	    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);

	    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, double kernel_value, Eigen::VectorXd &grad);

		void gradBasisFunction(const Eigen::VectorXd &x, const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad);

		void gradInverseWeightPrior(size_t p, Eigen::MatrixXd & diSigmadp);

	    void set_loghyper(const Eigen::VectorXd &p);

	    void set_loghyper(const double p[]);

	    std::string to_string();


	protected:
	    virtual bool real_init();
	private:
	    double g(const Eigen::VectorXd & x1, const Eigen::VectorXd & x2, const Eigen::VectorXd & sigma);

	    void initializeMatrices();

	    /**
	     * Signal variance factor.
	     */
	    double c;

	    /**
	     * Length scales.
	     */
	    Eigen::VectorXd ell;

	    /**
	     * Inducing input matrix.
	     */
	    Eigen::MatrixXd U;

	    /**
	     * Corresponding length scales.
	     */
	    Eigen::MatrixXd Uell;

	    /**
	     * Squared noise.
	     */
	    double sn2;

	    /**
	     * Squared inducing input noise.
	     */
	    double snu2;

	    /**
	     * The matrix Upsi.
	     */
	    Eigen::MatrixXd Upsi;

	    /**
	     * Cholesky of Upsi.
	     */
	    Eigen::MatrixXd LUpsi;

	    /**
	     * Inverse of Upsi.
	     */
	    Eigen::MatrixXd iUpsi;

	    /**
	     * log(|iUpsi|)/2
	     */
	    double halfLogDetiUpsi;
};
}


#endif /* INCLUDE_BASIS_FUNCTIONS_BF_MULTI_SCALE_H_ */
