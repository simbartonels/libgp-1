// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "rprop.h"
#include "gp_utils.h"

#include <sys/time.h>
#include <cmath>

static double tic() {
	struct timeval now;
	gettimeofday(&now, NULL);
	double time_in_seconds = now.tv_sec + now.tv_usec / 1000000.0;
	return time_in_seconds;
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
	  if(lik == NAN)
		  break;
	  if (verbose) std::cout << i << " " << -lik << std::endl;
  }
  gp->covf().set_loghyper(best_params);
}

void RProp::maximize(AbstractGaussianProcess * gp, Eigen::MatrixXd & param_history){
	  int param_dim = gp->covf().get_param_dim();
	  assert(param_history.rows() == param_dim + 1);
	  param_history.fill(-1);
	  size_t n = param_history.cols();
	  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
	  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
	  Eigen::VectorXd params = gp->covf().get_loghyper();
	  Eigen::VectorXd best_params = params;
	  double best = log(0.0);

	  double start = tic();
	  for (size_t i=0; i<n; ++i){
		  double lik = step(gp, best, Delta, grad_old, params, best_params);
		  double t = tic() - start;
		  if(lik == NAN)
			  break;
		  param_history.col(i).tail(param_dim) = params;
		  param_history(0, i) = t;
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
