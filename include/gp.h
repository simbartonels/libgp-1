// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*! 
 *  
 *   \page licence Licensing
 *    
 *     libgp - Gaussian process library for Machine Learning
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __GP_H__
#define __GP_H__

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "abstract_gp.h"
#include "cov.h"
#include "sampleset.h"

namespace libgp {
  
  /** Gaussian process regression.
   *  @author Manuel Blum, Simon Bartels */
  class GaussianProcess : public AbstractGaussianProcess
  {
  public:
    
    /** Create and instance of GaussianProcess with given input dimensionality 
     *  and covariance function. */
	GaussianProcess (size_t input_dim, std::string covf_def) : AbstractGaussianProcess(input_dim, covf_def){};

    /** Create and instance of GaussianProcess from file. */
    GaussianProcess (const char * filename) : AbstractGaussianProcess(filename){};
    
  protected:
    double var_impl(const Eigen::VectorXd &x_star);

    void grad_var_impl(const Eigen::VectorXd & x, Eigen::VectorXd & grad);

    double log_likelihood_impl();

    Eigen::VectorXd log_likelihood_gradient_impl();

    /** Update test input and cache kernel vector. */
    void update_k_star(const Eigen::VectorXd &x_star);

    void update_alpha();

    /** Compute covariance matrix and perform cholesky decomposition. */
    void computeCholesky();

    void updateCholesky(const double x[], double y);


  };
}

#endif /* __GP_H__ */
