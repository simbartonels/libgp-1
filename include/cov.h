// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_H__
#define __COV_H__

#include <iostream>
#include <vector>
#include "sampleset.h"

#include <Eigen/Dense>

namespace libgp {

/** Covariance function base class.
 *  @author Manuel Blum
 *  @ingroup cov_group
 *  @todo implement more covariance functions */
class CovarianceFunction {
public:
	/** Constructor. */
	CovarianceFunction() {
	}
	;

	/** Destructor. */
	virtual ~CovarianceFunction() {
	}
	;

	/** Initialization method for atomic covariance functions.
	 *  @param input_dim dimensionality of the input vectors */
	virtual bool init(int input_dim) {
		return false;
	}
	;

	/** Initialization method for compound covariance functions.
	 *  @param input_dim dimensionality of the input vectors
	 *  @param first first covariance function of compound
	 *  @param second second covariance function of compound */
	virtual bool init(int input_dim, CovarianceFunction * first,
			CovarianceFunction * second) {
		return false;
	}
	;

	virtual bool init(int input_dim, int filter, CovarianceFunction * covf) {
		return false;
	}
	;

	/** Computes the covariance of two input vectors.
	 *  @param x1 first input vector
	 *  @param x2 second input vector
	 *  @return covariance of x1 and x2 */
	virtual double get(const Eigen::VectorXd &x1,
			const Eigen::VectorXd &x2) = 0;

	/** Covariance gradient of two input vectors with respect to the hyperparameters.
	 *  The other function is more efficient if implemented but since that is not always the case
	 *  this funcion is kept for consistency.
	 *  @param x1 first input vector
	 *  @param x2 second input vector
	 *  @param grad covariance gradient */
	virtual void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
			Eigen::VectorXd &grad) = 0;

	/** Covariance gradient of two input vectors with respect to the hyperparameters.
	 *  @param x1 first input vector
	 *  @param x2 second input vector
	 *  @param kernel_value the value of k(x1,x2)
	 *  @param grad covariance gradient */
	virtual void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
			double kernel_value, Eigen::VectorXd &g) {
		grad(x1, x2, g);
	}
	;

	/**
	 * Computes dk(x,z)/dx. NOTE: implementations have to check for the case x=z!
	 * TODO: Actually a function that returns d k(x,z)/d xd (i.e. w.r.t. a specific dimension) would be more efficient.
	 */
	virtual void grad_input(const Eigen::VectorXd & x, const Eigen::VectorXd & z, Eigen::VectorXd & grad){
		//TODO: make this method abstract and implement it for every cov function
		std::cerr << "grad_input not implemented!" << std::endl;
//		exit(-1);
		grad.setZero();
	};

	/**
	 * Writes the derivative of k(x, X) w.r.t. x into J^T where J^T is D times size(k(x,X)).
	 * NOTE: The result MAY NOT be comparable to grad_input. Basis functions may divert.
	 * NOTE: The size of kstar is not necessarily n.
	 */
	virtual void compute_dkdx(const Eigen::VectorXd & x,
			const Eigen::VectorXd & kstar, SampleSet * sampleSet, Eigen::MatrixXd & JT){
		//TODO: make this method abstract and implement it for every cov function
		std::cerr << "dkdx not implemented!" << std::endl;
//		exit(-1);
		JT.setZero();
	};

	/** Update parameter vector.
	 *  @param p new parameter vector */
	virtual void set_loghyper(const Eigen::VectorXd &p);

	/** Update parameter vector.
	 *  @param p new parameter vector */
	virtual void set_loghyper(const double p[]);

	/** Get number of parameters for this covariance function.
	 *  @return parameter vector dimensionality */
	virtual size_t get_param_dim();

	/** Get input dimensionality.
	 *  @return input dimensionality */
	size_t get_input_dim();

	/** Get log-hyperparameter of covariance function.
	 *  @return log-hyperparameter */
	virtual Eigen::VectorXd get_loghyper();

	/** Returns a string representation of this covariance function.
	 *  @return string containing the name of this covariance function */
	virtual std::string to_string() = 0;

	virtual std::string pretty_print_parameters(){
		return "pretty printing parameters not implemented";
	};

	/** Draw random target values from this covariance function for input X. */
	virtual Eigen::VectorXd draw_random_sample(Eigen::MatrixXd &X);

	bool loghyper_changed;

protected:
	/** Input dimensionality. */
	size_t input_dim;

	/** Size of parameter vector. */
	size_t param_dim;

	/** Parameter vector containing the log hyperparameters of the covariance function.
	 *  The number of necessary parameters is given in param_dim. */
	Eigen::VectorXd loghyper;

};

}

#endif /* __COV_H__ */

/** Covariance functions available for Gaussian process models. 
 *  There are atomic and composite covariance functions. 
 *  @defgroup cov_group Covariance Functions */
