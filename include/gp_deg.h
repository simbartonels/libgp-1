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

    virtual inline void llh_setup_other();
    virtual inline void llh_setup_Gamma();
    /**
     * Updates variables such as noise and n.
     */
    virtual inline void update_internal_variables();

	Eigen::MatrixXd Phi;
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

    double log_noise;
    double squared_noise;

  private:
    inline double getSigmaGradient(size_t i);
    inline double getNoiseGradient();

    //TODO: think about a way to make this constant
    //one possibility is to construct an inner degenerate gp class
    bool sigmaIsDiagonal;


	Eigen::VectorXd temp;
    Eigen::MatrixXd Lu;

	Eigen::VectorXd Phiy;

	/**
	 * (Phi*y)^T*alpha
	 */
	double PhiyAlpha;

	bool recompute_yy;

	/**
	 * y^T*y
	 */
	double yy;

	Eigen::MatrixXd diSigma;
	Eigen::MatrixXd Gamma;
	Eigen::MatrixXd iAPhi;
	Eigen::VectorXd phi_alpha_minus_y;
  };
}




#endif /* SOURCE_DIRECTORY__INCLUDE_GP_DEC_H_ */
