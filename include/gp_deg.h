// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__INCLUDE_GP_DEC_H_
#define SOURCE_DIRECTORY__INCLUDE_GP_DEC_H_

#include <cmath>
#include <Eigen/Dense>

#include "abstract_gp.h"
#include "cov.h"
#include "sampleset.h"
#include "basis_functions/IBasisFunction.h"

namespace libgp {

  /** Approximate Gaussian process regression for degenerate kernels.
   * See section about weight space view in "Gaussian Processes for Machine Learning"
   * by Rasmussen and Williams from 2006.
   *  @author Manuel Blum, Simon Bartels */
  class DegGaussianProcess : public AbstractGaussianProcess
  {
  public:

    /** Create and instance of GaussianProcess with given input dimensionality
     *  and covariance function. */
	  DegGaussianProcess (size_t input_dim, std::string covf_def, size_t num_basisf, std::string basisf_def);
	  virtual ~DegGaussianProcess();

  protected:
    double var_impl(const Eigen::VectorXd &x_star);

    double log_likelihood_impl();

    Eigen::VectorXd log_likelihood_gradient_impl();

    /** Update test input and cache kernel vector. */
    void update_k_star(const Eigen::VectorXd &x_star);

    void update_alpha();

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void computeCholesky();

    void updateCholesky(const double x[], double y);

    /**
     * Computes the inverse of LL^T.
     * Unfortunately it seems necessary to compute the gradients.
     * Calls llh_setup_other().
     */
    virtual inline void llh_setup_Gamma();

    /**
     * Solves LL^T for Phi and other stuff that is not necessary for Solin's Laplace Approximation.
     */
    virtual inline void llh_setup_other();
    /**
     * Updates variables such as noise and n.
     */
    virtual inline void update_internal_variables();

    /**
     * Contains phi(X).
     */
	Eigen::MatrixXd Phi;

	/**
	 * Temporary variable for d Phi(X) / d theta_i.
	 */
	Eigen::MatrixXd dPhidi;

    /*
     * Convenience pointer that just points to cf.
     */
    IBasisFunction * bf;

    /**
     * The number of data points.
     */
    size_t n;

    /**
     * Number of basis functions.
     */
    size_t M;

    /**
     * Contains the log noise.
     */
    double log_noise;

    /**
     * Contains the squared noise.
     */
    double squared_noise;

  private:
    /**
     * Computes all contributions of Sigma to the derivative of the log-likelihood with respect
     * to parameter number i.
     * @param i
     * 	the parameter number
     * @returns
     *  contribution of Sigma to the gradient of the log-likelihood
     */
    inline double getSigmaGradient(size_t i);

    /**
     * Computes the noise gradient for the log-likelihood.
     * @returns
     *  Gradient of the log-likelihood with respect to the noise.
     */
    inline double getNoiseGradient();

    //TODO: think about a way to make this constant
    //one possibility is to construct an inner degenerate gp class
    bool sigmaIsDiagonal;


    /**
     * Temporary vector of size M. Used in various locations.
     */
	Eigen::VectorXd temp;

	/**
	 * Cholesky decomposition of Phi*Phi^T.
	 */
    Eigen::MatrixXd Lu;

    /**
     * Contains Phi*y.
     */
	Eigen::VectorXd Phiy;

	/**
	 * (Phi*y)^T*alpha
	 */
	double PhiyAlpha;

	/**
	 * Flag that signals whether new data points have been added or not.
	 */
	bool recompute_yy;

	/**
	 * y^T*y
	 */
	double yy;

	/**
	 * Space for d Sigma / d theta_i.
	 */
	Eigen::MatrixXd diSigma;

	/**
	 * Contains the inverse of LL^T.
	 */
	Eigen::MatrixXd Gamma;

	/**
	 * Contains LL^T solved for Phi.
	 */
	Eigen::MatrixXd iAPhi;

	/**
	 * Contains Phi*alpha-y.
	 */
	Eigen::VectorXd phi_alpha_minus_y;
  };
}




#endif /* SOURCE_DIRECTORY__INCLUDE_GP_DEC_H_ */
