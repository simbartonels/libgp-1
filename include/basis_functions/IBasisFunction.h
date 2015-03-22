// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_
#define SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_

#include "cov.h"
#include "sampleset.h"

#include "time.h"

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
		std::cerr
				<< "IBasisFunction: Wrong initialization method for basis functions!"
				<< std::endl;
		return false;
	}

	/** Initialization method for atomic basis functions. Randomly initializes a seed and calls
	 * init(M, wrappeCov, seed).
	 *  @param M the number of basis functions
	 *  @param wrappedCovFunc the wrapped covariance function
	 *  @return true if initialization was successful.
	 */
	bool init(size_t M, CovarianceFunction * wrappedCovFunc) {
		size_t seed = (size_t) time(0);
		return init(M, wrappedCovFunc, seed);
	}
	;

	/** Initialization method for atomic basis functions.
	 *  @param M the number of basis functions
	 *  @param wrappedCovFunc the wrapped covariance function
	 *  @param seed for random numbers
	 *  @return true if initialization was successful.
	 */
	bool init(size_t M, CovarianceFunction * wrappedCovFunc, size_t seed){
		//TODO (Simon): refactor. this is actually already a lot of functionality for a header file
		input_dim = wrappedCovFunc->get_input_dim();

		this->M = M;
		this->seed = seed;
		cov = wrappedCovFunc;
		//+1 for noise
		param_dim = get_param_dim_without_noise(input_dim, M) + 1;
		loghyper.resize(param_dim);
		return real_init();
	}

	/**
	 * Writes k(x, x) into diag where diag is of size sampleSet->size().
	 * ATTENTION: For efficiency this method is to be overwritten!
	 * @param sampleSet the sample set
	 * @param diag the output vector
	 */
	virtual void putDiagWrapped(SampleSet * sampleSet, Eigen::VectorXd& diag){
		size_t n = sampleSet->size();
		assert(diag.size() == n);
		for(size_t i = 0; i < n; i++)
			diag(i) = getWrappedKernelValue(sampleSet->x(i), sampleSet->x(i));
	};

	/**
	 * Computes the values of all basis functions for a given vector.
	 * The underlying input distribution and the number of basis
	 * functions should be determined during construction.
	 * @param x input vector
	 * @return the vector of basis function values
	 */
	//TODO: this is kinda bad since it requires n allocations multiple times!!!
	//but it's equally bad for all implementations so nothing to be too concerned about for my thesis
	virtual Eigen::VectorXd computeBasisFunctionVector(
			const Eigen::VectorXd &x) = 0;

	virtual void gradBasisFunction(SampleSet * sampleSet, const Eigen::MatrixXd & Phi, size_t p, Eigen::MatrixXd & Grad) = 0;

	/**
	 * Computes the derivative of a basis function vector phi(x) with respect to parameter i.
	 * Implementing classes MAY assume for the first call that grad is 0 initialized.
	 * @param x the input to phi
	 * @param phi phi(x)
	 * @param p the number of the parameter
	 * @param grad where to put the result
	 */
	virtual void gradBasisFunction(const Eigen::VectorXd &x,
			const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad){
		SampleSet * sampleSet = new SampleSet(x.size());
		sampleSet->add(x.data(), 0.);
//		Eigen::Map<Eigen::MatrixXd> Phi(phi.data(), phi.rows(), phi.cols());
		Eigen::MatrixXd Phi = phi;
		Eigen::MatrixXd Grad = grad;
		gradBasisFunction(sampleSet, Phi, p, Grad);
		grad = Grad.col(0);
		delete sampleSet;
	};

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
	virtual const Eigen::MatrixXd & getSigma() = 0;


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
	virtual const Eigen::MatrixXd & getInverseOfSigma() = 0;

	/**
	 * Returns the Cholesky of Sigma.
	 * TODO: implement and allow override
	 */
	virtual const Eigen::MatrixXd & getCholeskyOfInvertedSigma() = 0;

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
		//the last sum() is to convince Eigen that we can return a double.
		return (computeBasisFunctionVector(x1).transpose() * getSigma() * computeBasisFunctionVector(x2)).sum();
	}

	/**
	 * Returns the gradient of diagonal of the wrapped kernel matrix, i.e. dk(x,x)/dt.
	 *
	 * ATTENTION: The method is highly inefficient and should be overwritten!
	 *
	 * ASSUMPTION: diagK_i == wrappedCov(sampleset->x(i), sampleset->x(i))
	 *
	 * @param sampleset Sampleset of all datapoints.
	 * @param diagK The diagonal of the kernel matrix.
	 * @param parameter The parameter for which to compute the gradient.
	 * @param gradient Where to write the output.
	 */
	virtual void gradDiagWrapped(SampleSet * sampleset, const Eigen::VectorXd & diagK, size_t parameter, Eigen::VectorXd & gradient){
		//highly inefficient but this method should be overwritten anyways...
		assert(diagK.size() == gradient.size());
		size_t n = sampleset->size();
		Eigen::VectorXd t(param_dim);
		for (size_t j = 0; j < n; j++) {
			double temp = diagK(j);
			grad(sampleset->x(j), sampleset->x(j), temp, t);
			gradient(j) = t(parameter);
		}
	}

	/**
	 * Returns true if gradDiagWrapped(..., parameter, ...) is all zeros.
	 * @param parameter The number of the parameter that is optimized.
	 */
	virtual bool gradDiagWrappedIsNull(size_t parameter) = 0;

	/**
	 * Returns the noise on a log scale.
	 * ATTENTION: The noise MUST be the last parameter!
	 * This is necessary for hyper-parameter optimization for degenerate kernels.
	 * Unfortunately, there seems to be no easy way to hide this parameter.
	 */
	virtual double getLogNoise(){
		return loghyper(get_param_dim() - 1);
	}

	void set_loghyper(const Eigen::VectorXd &p){
		CovarianceFunction::set_loghyper(p);
		//TODO: think about noise!
		log_hyper_updated(p);
	};

	//TODO: Make this protected!
	/**
	 * Pointer to the wrapped covariance function.
	 */
	CovarianceFunction * cov;

protected:
	/**
	 * Performs the actual initialization.
	 */
	virtual bool real_init() = 0;

	/**
	 * Notifies the basis function that new hyper-parameters arrived.
	 */
	virtual void log_hyper_updated(const Eigen::VectorXd &p) = 0;

	/**
	 * Returns the number of hyper-parameters WITHOUT noise given the input dimensionality
	 * and the required number of basis functions.
	 * The number of parameters should depend only on the two provided arguments. This function
	 * is called before real_init.
	 * Functions may assume that cov is already set and initialized.
	 * @param input_dim The input dimensionality.
	 * @param num_basis_functions The number of basis functions (M).
	 * @return The number of hyper-parameters.
	 */
	virtual size_t get_param_dim_without_noise(size_t input_dim, size_t num_basis_functions) = 0;

	size_t M;

	size_t seed;
};
}

#endif /* SOURCE_DIRECTORY__INCLUDE_IBASISFUNCTION_H_ */
