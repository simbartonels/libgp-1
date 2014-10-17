// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "fic_gp.h"
#include "cov_factory.h"
#include "basis_functions/basisf_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {

  const double log2pi = log(2*M_PI);

  //TODO: find a way to get the value from the super class
  const double initial_L_size = 1000;

  FICGaussianProcess::FICGaussianProcess(size_t input_dim, std::string covf_def, size_t num_basisf, std::string basisf_def)
   :AbstractGaussianProcess(input_dim, covf_def){
	  BasisFFactory factory;
	  //wrap initialized covariance function with basis function
	  cf = factory.createBasisFunction(basisf_def, num_basisf, cf);
	  cf->loghyper_changed = 0;
	  bf = (IBasisFunction *) cf;
  }

  FICGaussianProcess::~FICGaussianProcess(){
//	  delete Lu;
//	  delete Luu;
  }

  double FICGaussianProcess::var_impl(const Eigen::VectorXd x_star){
	  return bf->getWrappedKernelValue(x_star, x_star)
			  + k_star.transpose()*L*k_star;
  }

  void FICGaussianProcess::computeCholesky()
  {
	/*
	 * This method does not compute the Cholesky in the same sense as
	 * the GaussianProcess class does. Here the same thing happens as
	 * in infFITC.m from the gpml toolbox by Rasmussen and Nikisch. The
	 * method computeCholesky is kept for abstraction reasons.
	 */
	size_t M = bf->getNumberOfBasisFunctions();
	size_t n = sampleset->size();
    if (M > L.rows()){
    	L.resize(M, M);
    	Lu.resize(M, M);
    	Luu.resize(M, M);
    	V.resize(M, n);
    }
    if(n > isqrtgamma.rows()){
    	isqrtgamma.resize(n);
    	V.resize(M, n);
    }
    //corresponds to Ku in infFITC
    Eigen::MatrixXd Phi(M, n);
    //corresponds to diagK in infFITC
    Eigen::VectorXd k(n);
    for(size_t i = 0; i < n; i++) {
      Eigen::VectorXd xi = sampleset->x(i);
      Eigen::VectorXd phi = bf->computeBasisFunctionVector(xi);
      //TODO: is there a faster operation?
      for(size_t j = 0; j < M; j++) {
    	Phi(j, i) = phi(j);
      }
      //TODO: rethink the design here
      //it might be better to seperate basis functions and kernels
      k(i) = bf->getWrappedKernelValue(xi, xi);
    }
	double snu2 = 0; //1e-6*sn2 //hard coded inducing inputs noise
    Luu = bf->getCholeskyOfInverseWeightPrior();
    V = Luu.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(Phi);
    //noise is already added in k
    isqrtgamma = k + (V.transpose()*V).diagonal();
    //isqrtgamma = Eigen::VectorXd::Ones(n).cwiseQuotient(isqrtgamma);
    isqrtgamma.array() = 1/isqrtgamma.array().sqrt();
    V = V * isqrtgamma.asDiagonal();
    // TODO: is it possible to use the self adjoint view here?
    Lu = V * V.transpose() + Eigen::MatrixXd::Identity(M, M);
    Lu.topLeftCorner(M, M) = Lu.llt().matrixL();
	Eigen::MatrixXd iUpsi = bf->getWeightPrior();
    L = Lu * Luu;
    L.topLeftCorner(M, M) = L.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(M, M));
    L = L.transpose() * L;
    L = L - iUpsi;
  }

  void FICGaussianProcess::updateCholesky(const double x[], double y){
	  //Do nothing and just recompute everything.
	  //TODO: might be a slow down in applications!
	  cf->loghyper_changed = true;
  }

  void FICGaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    k_star.resize(bf->getNumberOfBasisFunctions());
    k_star = bf->computeBasisFunctionVector(x_star);
  }

  void FICGaussianProcess::update_alpha()
  {
	    size_t M = bf->getNumberOfBasisFunctions();
	    //TODO: can we avoid this?
	    alpha.resize(M);
	    size_t n = sampleset->size();
	    Eigen::VectorXd r(n);
	    // Map target values to VectorXd
	    const std::vector<double>& targets = sampleset->y();
	    Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	    r = y.array() * isqrtgamma.array();
	    Eigen::VectorXd beta = V * r;
		Lu.transpose().topLeftCorner(M, M).triangularView<Eigen::Upper>().solveInPlace(beta);
		//alpha = Luu\(Lu\be)
		alpha = Lu.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(beta);
		Luu.topLeftCorner(M, M).triangularView<Eigen::Lower>().solveInPlace(alpha);
  }

  double FICGaussianProcess::log_likelihood_impl() {
  	return 0;
  }

  Eigen::VectorXd FICGaussianProcess::log_likelihood_gradient_impl() {
  	return Eigen::VectorXd::Zero(bf->get_param_dim());
  }
}
