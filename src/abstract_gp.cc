// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "abstract_gp.h"
#include "cov_factory.h"

#include <iostream>

namespace libgp {

const double initial_L_size = 1000;
AbstractGaussianProcess::AbstractGaussianProcess (size_t input_dim, std::string covf_def)
{
  // set input dimensionality
  this->input_dim = input_dim;
  // create covariance function
  CovFactory factory;
  cf = factory.create(input_dim, covf_def);
  cf->loghyper_changed = 0;
  sampleset = new SampleSet(input_dim);
  L.resize(initial_L_size, initial_L_size);
}

AbstractGaussianProcess::~AbstractGaussianProcess ()
{
  // free memory
  delete sampleset;
  delete cf;
}

double AbstractGaussianProcess::f(const double x[])
{
  if (sampleset->empty()) return 0;
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_k_star(x_star);
  return k_star.dot(alpha);
}


double AbstractGaussianProcess::var(const double x[])
{
  if (sampleset->empty()) return 0;
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_k_star(x_star);
  return var_impl(x_star);
}

double AbstractGaussianProcess::log_likelihood()
{
  compute();
  return log_likelihood_impl();
}

Eigen::VectorXd AbstractGaussianProcess::log_likelihood_gradient()
{
  compute();
  return log_likelihood_gradient_impl();
}

void AbstractGaussianProcess::add_pattern(const double x[], double y)
{
  //std::cout<< L.rows() << std::endl;
  sampleset->add(x, y);
  alpha_needs_update = true;
  updateCholesky(x, y);
}

bool AbstractGaussianProcess::set_y(size_t i, double y)
{
  if(sampleset->set_y(i,y)) {
    alpha_needs_update = true;
    return 1;
  }
  return false;
}

size_t AbstractGaussianProcess::get_sampleset_size()
{
  return sampleset->size();
}

void AbstractGaussianProcess::clear_sampleset()
{
  sampleset->clear();
}


CovarianceFunction & AbstractGaussianProcess::covf()
{
  return *cf;
}

size_t AbstractGaussianProcess::get_input_dim()
{
  return input_dim;
}


void AbstractGaussianProcess::compute(){
	// can previously computed values be used?
	if (cf->loghyper_changed){
		cf->loghyper_changed = false;
		alpha_needs_update = true;
		computeCholesky();
	}

    // can previously computed values be used?
    if (alpha_needs_update){
    	alpha_needs_update = false;
    	update_alpha();
    }
}

}

