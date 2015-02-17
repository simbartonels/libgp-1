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
#ifndef __ABSTRACT_GP_H_
#define __ABSTRACT_GP_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "cov.h"
#include "sampleset.h"

namespace libgp {

  /** Gaussian process interface.
   *  @author Simon Bartels, Manuel Blum */
  class AbstractGaussianProcess
  {
  public:
	/** Create and instance of GaussianProcess with given input dimensionality
	 *  and covariance function. */
	AbstractGaussianProcess (size_t input_dim, std::string covf_def);

    /** Create and instance of GaussianProcess from file. */
	AbstractGaussianProcess (const char * filename);

    virtual ~AbstractGaussianProcess();

    /** Write current gp model to file. */
    void write(const char * filename);

    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    double f(const double x[]);

    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    double var(const double x[]);

    /**
     * Returns the POSITIVE log-likelihood.
     */
    double log_likelihood();

    Eigen::VectorXd log_likelihood_gradient();

    /** Add input-output-pair to sample set.
     *  Add a copy of the given input-output-pair to sample set.
     *  @param x input array
     *  @param y output value
     */
    void add_pattern(const double x[], double y);
    void add_pattern(const Eigen::VectorXd & x, double y);


    bool set_y(size_t i, double y);

    /** Get number of samples in the training set. */
    size_t get_sampleset_size();

    /** Clear sample set and free memory. */
    void clear_sampleset();

    /** Get reference on currently used covariance function. */
    CovarianceFunction & covf();

    /** Get input vector dimensionality. */
    size_t get_input_dim();

    /**
     * Returns a copy of the matrix L.
     */
    Eigen::MatrixXd getL();

    /**
     * Returns a copy of the vector alpha.
     */
    Eigen::VectorXd getAlpha();

  protected:
    virtual double var_impl(const Eigen::VectorXd &x_star) = 0;

    virtual double log_likelihood_impl() = 0;

    virtual Eigen::VectorXd log_likelihood_gradient_impl() = 0;

    /** Update test input and cache kernel vector. */
    virtual void update_k_star(const Eigen::VectorXd &x_star) = 0;

    virtual void update_alpha() = 0;

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void computeCholesky() = 0;

    virtual void updateCholesky(const double x[], double y) = 0;

    /** The covariance function of this Gaussian process. */
    CovarianceFunction * cf;

    /** The training sample set. */
    SampleSet * sampleset;

    /** Alpha is cached for performance. */
    Eigen::VectorXd alpha;

    /** Last test kernel vector. */
    Eigen::VectorXd k_star;

    /** Linear solver used to invert the covariance matrix. */
//    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::MatrixXd L;

    /** Input vector dimensionality. */
    size_t input_dim;

  private:
    /** Internally called before any actual computations like
     * computeCholesky() or  updateAlpha() are called. Checks
     * if these updates are actually necessary. */
    void compute();

    bool alpha_needs_update;

  };
}

#endif /* SOURCE_DIRECTORY__INCLUDE_ABSTRACT_GP_H_ */
