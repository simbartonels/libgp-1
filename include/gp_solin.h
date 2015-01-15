// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef SOURCE_DIRECTORY__INCLUDE_GP_SOLIN_H_
#define SOURCE_DIRECTORY__INCLUDE_GP_SOLIN_H_

#include <cmath>
#include <Eigen/Dense>

#include "gp_deg.h"
#include "cov.h"
#include "sampleset.h"
#include "basis_functions/IBasisFunction.h"

namespace libgp {

//TODO: add to cmake list
//TODO: might need to make teh cholesky a protected field (though isn't it protected from the abstract_gp anyway?)

  /** Derived class that implements "Hilbert Space Methods for Reduced Rank Gaussian Process Regression" by Solin and Särkkä from 2014.
   *  The gp_deg class does not allow to use all the advantages of this approach and therefore an extra implementation is necessary.
   *  @author Manuel Blum, Simon Bartels */
  class SolinGaussianProcess : public DegGaussianProcess
  {
  public:

    /** Create and instance of GaussianProcess with given input dimensionality
     *  and covariance function. */
	  SolinGaussianProcess (size_t input_dim, std::string covf_def, size_t num_basisf, std::string basisf_def);
	  virtual ~SolinGaussianProcess();

  protected:
    virtual void computeCholesky();

  protected:
    virtual inline void llh_setup();

  private:
    Eigen::MatrixXd PhiPhi;
  };
}
#endif /* SOURCE_DIRECTORY__INCLUDE_GP_SOLIN_H_ */
