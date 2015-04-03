// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_sum.h"
#include "cmath"

namespace libgp
{
  
  CovSum::CovSum()
  {
  }
  
  CovSum::~CovSum()
  {
    delete first;
    delete second;
  }
  
  bool CovSum::init(int n, CovarianceFunction * first, CovarianceFunction * second)
  {
    this->input_dim = n;
    this->first = first;
    this->second = second;
    param_dim_first = first->get_param_dim();
    param_dim_second = second->get_param_dim();
    param_dim = param_dim_first + param_dim_second;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovSum::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    return first->get(x1, x2) + second->get(x1, x2);
  }
  
  void CovSum::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd grad_first(param_dim_first);
    Eigen::VectorXd grad_second(param_dim_second);
    first->grad(x1, x2, grad_first);
    second->grad(x1, x2, grad_second);
    grad.head(param_dim_first) = grad_first;
    grad.tail(param_dim_second) = grad_second;
  }
  
  void CovSum::grad_input(const Eigen::VectorXd & x, const Eigen::VectorXd & z, Eigen::VectorXd & grad){
	  first->grad_input(x, z, grad);
	  //TODO: this allocation is inefficient
	  Eigen::VectorXd grad2(input_dim);
	  second->grad_input(x, z, grad2);
	  grad += grad2;
  }

  double CovSum::grad_input_d(double xd, double zd, double k, size_t d){
	  return first->grad_input_d(xd, zd, k, d)+second->grad_input_d(xd, zd, k, d);
  }

  void CovSum::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    first->set_loghyper(p.head(param_dim_first));
    second->set_loghyper(p.tail(param_dim_second));
  }
  
  std::string CovSum::to_string()
  {
    return "CovSum("+first->to_string()+", "+second->to_string()+")";
  }
}
