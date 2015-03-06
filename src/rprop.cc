// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "rprop.h"
#include "gp_utils.h"
#define NOMINMAX //to prevent windows headers from introducing MIN MAX macros
#if defined(WIN32) || defined(WIN64) || defined(_WIN32) || defined(_WIN64)
	#include "winsock2.h"
#else
	#include <sys/time.h>
#endif
#include <cmath>

#ifdef _MSC_VER
#include <float.h>
#define INFINITY (DBL_MAX+DBL_MAX)
#define NAN (INFINITY-INFINITY)
static bool isnan(double d){
	return _isnan(d);
}
//#else
//static bool isnan(double d){
//	return std::isnan(d);
//}
#endif

static double tic() {
#if defined(WIN32) || defined(WIN64) || defined(_WIN32) || defined(_WIN64)
	return GetTickCount() / 1000.0;
#else
	struct timeval now;
	gettimeofday(&now, NULL);
	double time_in_seconds = now.tv_sec + now.tv_usec / 1000000.0;
	return time_in_seconds;
#endif
}

namespace libgp {

void RProp::init(double eps_stop, double Delta0, double Deltamin, double Deltamax, double etaminus, double etaplus) 
{
  this->Delta0   = Delta0;
  this->Deltamin = Deltamin;
  this->Deltamax = Deltamax;
  this->etaminus = etaminus;
  this->etaplus  = etaplus;
  this->eps_stop = eps_stop;

}

void RProp::maximize(AbstractGaussianProcess * gp, size_t n, bool verbose)
{
  int param_dim = gp->covf().get_param_dim();
  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
  Eigen::VectorXd params = gp->covf().get_loghyper();
  Eigen::VectorXd best_params = params;
  double best = log(0.0);

  for (size_t i=0; i<n; ++i) {
	  double lik = step(gp, best, Delta, grad_old, params, best_params);
	  if(isnan(lik))
		  break;
	  if (verbose) std::cout << i << " " << -lik << std::endl;
  }
  gp->covf().set_loghyper(best_params);
}

void RProp::maximize(AbstractGaussianProcess * gp, const Eigen::MatrixXd & testX,
		Eigen::VectorXd & times, Eigen::MatrixXd & param_history, Eigen::MatrixXd & meanY,
		Eigen::MatrixXd & varY, Eigen::VectorXd & nllh){
	  int param_dim = gp->covf().get_param_dim();
	  size_t iters = times.size();
	  size_t input_dim = gp->get_input_dim();
	  size_t n = gp->get_sampleset_size();
	  size_t n_test = testX.rows();
	  assert(testX.cols() == input_dim);
	  assert(param_history.rows() == param_dim);
	  assert(param_history.cols() == iters);
	  assert(meanY.rows() == n_test);
	  assert(meanY.cols() == iters);
	  assert(varY.size() == meanY.size());
	  assert(nllh.size() == iters);
	  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
	  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
	  Eigen::VectorXd params = gp->covf().get_loghyper();
	  Eigen::VectorXd best_params = params;
	  double best = log(0.0);

	  double start = tic();
	  for (size_t i=0; i<iters; ++i){
		  double lik = step(gp, best, Delta, grad_old, params, best_params);
		  double t = tic() - start;
		  if(isnan(lik))
			  break;
		  std::cout << i << " " << -lik << std::endl;
		  param_history.col(i) = params;
		  times(i) = t;
		  nllh(i) = -best; //best has been updated in step
		  if(lik == best){
			  //compute mean and variance for all test inputs
			  for(size_t j = 0; j < testX.rows(); j++){
				  meanY(j, i) = gp->f(testX.row(j));
				  varY(j, i) = gp->var(testX.row(j));
			  }
		  }
		  else{
			  //just copy results from last prediction
			  meanY.col(i) = meanY.col(i-1);
			  varY.col(i) = varY.col(i-1);
		  }
	  }
	  gp->covf().set_loghyper(best_params);
}

inline double RProp::step(AbstractGaussianProcess * gp, double & best, Eigen::VectorXd & Delta, Eigen::VectorXd & grad_old, Eigen::VectorXd & params, Eigen::VectorXd & best_params){
	Eigen::VectorXd grad = -gp->log_likelihood_gradient();
    grad_old = grad_old.cwiseProduct(grad);
    for (int j=0; j<grad_old.size(); ++j) {
      if (grad_old(j) > 0) {
        Delta(j) = std::min(Delta(j)*etaplus, Deltamax);        
      } else if (grad_old(j) < 0) {
        Delta(j) = std::max(Delta(j)*etaminus, Deltamin);
        grad(j) = 0;
      } 
      params(j) += -Utils::sign(grad(j)) * Delta(j);
    }
    grad_old = grad;
    if (grad_old.norm() < eps_stop) return NAN;
    gp->covf().set_loghyper(params);
    double lik = gp->log_likelihood();
    if (lik > best) {
      best = lik;
      best_params = params;
    }
    return lik;
}

}
