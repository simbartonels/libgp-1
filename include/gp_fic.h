// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__SRC_GP_APPR_H_
#define SOURCE_DIRECTORY__SRC_GP_APPR_H_

#include <cmath>
#include <Eigen/Dense>

#include "abstract_gp.h"
#include "cov.h"
#include "sampleset.h"
#include "basis_functions/IBasisFunction.h"

namespace libgp {

  /**
   * Approximate Gaussian process regression using FIC.
   * See "Approximation Methods for Gaussian Processes" by Quinonero-Candela,
   * Rasmussen and Williams from 2007. The implementation follows the framework written
   * by Chalupka. See "A Framework for Evaluating Approximation Methods for Gaussian Process
   * Regression" by Chalupka, Williams and Murray from 2013.
   *
   *  @author Manuel Blum, Simon Bartels
   */
  class FICGaussianProcess : public AbstractGaussianProcess
  {
  public:

    /** Create and instance of GaussianProcess with given input dimensionality
     *  and covariance function. */
	  FICGaussianProcess (size_t input_dim, std::string covf_def, size_t num_basisf, std::string basisf_def);// : AbstractGaussianProcess(input_dim, covf_def){};
    /** Create and instance of GaussianProcess from file. */
//    FICGaussianProcess (const char * filename) : AbstractGaussianProcess(filename) {};

	  virtual ~FICGaussianProcess();

  protected:
    double var_impl(const Eigen::VectorXd &x_star);

    void var_grad_impl(const Eigen::VectorXd & x, Eigen::VectorXd & grad);

    double log_likelihood_impl();

    Eigen::VectorXd log_likelihood_gradient_impl();

    /** Update test input and cache kernel vector. */
    void update_k_star(const Eigen::VectorXd &x_star);

    void update_alpha();

    /** Compute covariance matrix and perform cholesky decomposition. */
    void computeCholesky();

    void updateCholesky(const double x[], double y);

  private:
	/**
	 * Corresponds to diagK in infFITC.
	 */
    Eigen::VectorXd k;

	/**
	 * Corresponds to Ku in infFITC.
	 */
	Eigen::MatrixXd Phi;
    Eigen::MatrixXd Lu;
    Eigen::VectorXd dg;
	Eigen::VectorXd al;
	Eigen::VectorXd alSqrd;
    Eigen::VectorXd isqrtgamma;
    Eigen::MatrixXd V;
    Eigen::MatrixXd B;
    Eigen::MatrixXd W;
    Eigen::VectorXd w;
    Eigen::MatrixXd Wdg;
    Eigen::MatrixXd R;
    Eigen::VectorXd v;
    Eigen::MatrixXd BWdg;
    Eigen::VectorXd WdgSum;

    Eigen::VectorXd ddiagK;
	Eigen::MatrixXd dKui;
	Eigen::MatrixXd dKuui;

	Eigen::VectorXd r;
	Eigen::VectorXd beta;

	Eigen::VectorXd temp;

	/**
	 * Temporary matrix to save the product of Luu and Lu.
	 */
	Eigen::MatrixXd LuuLu;

    /**
     * Convenience pointer that just points to the covariance function cf in the super class.
     */
    IBasisFunction * bf;

    /**
     * Number of basis functions.
     */
    size_t M;
  };
}




#endif /* SOURCE_DIRECTORY__SRC_GP_APPR_H_ */
