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

		Eigen::MatrixXd getInverseOfSigma();

		Eigen::MatrixXd getCholeskyOfInvertedSigma();

		Eigen::MatrixXd getSigma();

		bool sigmaIsDiagonal(){
			return false;
		};

		double getLogDeterminantOfSigma();

		/**
		 * Parent is overwritten since this kernel does not exactly wrap the ARDse.
		 */
		double getWrappedKernelValue(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);

	    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);

	    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, double kernel_value, Eigen::VectorXd &grad);

		void gradBasisFunction(const Eigen::VectorXd &x, const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad);

		bool gradBasisFunctionIsNull(size_t p){
			//TODO: implement in cc file
			return false;
		};

		void gradiSigma(size_t p, Eigen::MatrixXd & diSigmadp);

		bool gradiSigmaIsNull(size_t p){
			//TODO: implement in cc file
			return false;
		};

	    std::string to_string();


	protected:
	    virtual bool real_init();

	    void log_hyper_updated(const Eigen::VectorXd & p);

	    size_t get_param_dim_without_noise(size_t input_dim, size_t num_basis_functions);
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
	     * log(|Upsi|)/2
	     */
	    double halfLogDetiUpsi;
};
}


#endif /* INCLUDE_BASIS_FUNCTIONS_BF_MULTI_SCALE_H_ */
