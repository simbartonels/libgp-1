// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_MULTI_SCALE_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_MULTI_SCALE_H_

#include "IBasisFunction.h"

namespace libgp{
class MultiScale : public IBasisFunction{
public:
		void putDiagWrapped(SampleSet * sampleSet, Eigen::VectorXd& diag);

		Eigen::VectorXd computeBasisFunctionVector(const Eigen::VectorXd &x);

		const Eigen::MatrixXd & getInverseOfSigma();

		const Eigen::MatrixXd & getCholeskyOfInvertedSigma();

		const Eigen::MatrixXd & getSigma();

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

	    void gradDiagWrapped(SampleSet * sampleset, const Eigen::VectorXd & diagK, size_t parameter, Eigen::VectorXd & gradient);

	    bool gradDiagWrappedIsNull(size_t parameter);

		void gradBasisFunction(const Eigen::VectorXd &x, const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad);

		void gradBasisFunction(SampleSet * sampleSet, const Eigen::MatrixXd &Phi, size_t p, Eigen::MatrixXd &Grad);

		bool gradBasisFunctionIsNull(size_t p);

		void gradiSigma(size_t p, Eigen::MatrixXd & diSigmadp);

		bool gradiSigmaIsNull(size_t p);

	    std::string to_string();


	protected:
	    virtual bool real_init();

	    void log_hyper_updated(const Eigen::VectorXd & p);

	    size_t get_param_dim_without_noise(size_t input_dim, size_t num_basis_functions);
	private:
	    inline double g(const Eigen::VectorXd & x1, const Eigen::VectorXd & x2, const Eigen::VectorXd & sigma);

	    void initializeMatrices();

	    /**
	     * Given a parameter number and whether the parameter corresponds to a length scale or an
	     * inducing point this function sets previous_m and previous_d accordingly. I.e. it sets
	     * the number of the length scale / inducing point and the dimension as to access U and Uell.
	     * @param p the number of the parameter
	     * @param lengthScaleDerivative whether the parameter corresponds to an inducing length scale
	     * 	or an inducing point
	     */
	    void inline setPreviousNumberAndDimensionForParameter(size_t p,
	    		bool lengthScaleDerivative);

	    /**
	     * Signal variance factor.
	     */
	    double c;

	    /**
	     * Contains c/sqrt(|2*pi*diag(ell)|) + sn2.
	     */
	    double c_over_ell_det;

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
	     * Contains the products of the length scales.
	     */
	    Eigen::VectorXd factors;

	    /**
	     * Temporary vector that contains x-z.
	     */
	    Eigen::VectorXd delta;

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

	    /**
	     * Vector used in the gradient computation of iSigma.
	     */
		Eigen::VectorXd temp;
		Eigen::VectorXd UpsiCol;

		/**
		 * Temporary vector of size input_dim. Used in initializeMatrices.
		 */
		Eigen::VectorXd temp_input_dim;

		/**
		 * Contains the parameter number of the last computed gradient.
		 */
		size_t previous_p;

		/**
		 * Contains the number of the basis vector or length scale of the last computed gradient.
		 */
		size_t previous_m;

		/**
		 * Contains the dimension of the basis vector or length scale of the last computed gradient.
		 */
		size_t previous_d;
};
}


#endif /* INCLUDE_BASIS_FUNCTIONS_BF_MULTI_SCALE_H_ */
