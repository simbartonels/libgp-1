// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "rprop.h"
#include "gp_utils.h"

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

void RProp::maximize(GaussianProcess * gp, size_t n, bool verbose)
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

Eigen::MatrixXd RProp::maximize(GaussianProcess * gp, size_t n){
	  int param_dim = gp->covf().get_param_dim();
	  Eigen::MatrixXd param_history(n, param_dim);
	  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
	  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
	  Eigen::VectorXd params = gp->covf().get_loghyper();
	  Eigen::VectorXd best_params = params;
	  double best = log(0.0);

	  for (size_t i=0; i<n; ++i){
		  double lik = step(gp, best, Delta, grad_old, params, best_params);
		  if(lik == NAN)
			  break;
		  param_history.row(i) = params;
	  }
	  gp->covf().set_loghyper(best_params);
	  return param_history;
}

double RProp::step(GaussianProcess * gp, double & best, Eigen::VectorXd & Delta, Eigen::VectorXd & grad_old, Eigen::VectorXd & params, Eigen::VectorXd & best_params){
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
