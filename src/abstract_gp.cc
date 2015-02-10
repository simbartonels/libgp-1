// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "abstract_gp.h"
#include "cov_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>


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
  alpha_needs_update = false;
  sampleset = new SampleSet(input_dim);
  L.resize(initial_L_size, initial_L_size);
}

  AbstractGaussianProcess::AbstractGaussianProcess (const char * filename)
  {
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double * x = NULL;
    L.resize(initial_L_size, initial_L_size);
    while (infile.good()) {
      getline(infile, s);
      // ignore empty lines and comments
      if (s.length() != 0 && s.at(0) != '#') {
        std::stringstream ss(s);
        if (stage > 2) {
          ss >> y;
          for(size_t j = 0; j < input_dim; ++j) {
            ss >> x[j];
          }
          add_pattern(x, y);
        } else if (stage == 0) {
          ss >> input_dim;
          sampleset = new SampleSet(input_dim);
          x = new double[input_dim];
        } else if (stage == 1) {
          CovFactory factory;
          cf = factory.create(input_dim, s);
          cf->loghyper_changed = 0;
        } else if (stage == 2) {
          Eigen::VectorXd params(cf->get_param_dim());
          for (size_t j = 0; j<cf->get_param_dim(); ++j) {
            ss >> params[j];
          }
          cf->set_loghyper(params);
        }
        stage++;
      }
    }
    infile.close();
    if (stage < 3) {
      std::cerr << "fatal error while reading " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    delete [] x;
  }

AbstractGaussianProcess::~AbstractGaussianProcess ()
{
  // free memory
  delete sampleset;
  delete cf;
}

void AbstractGaussianProcess::write(const char * filename)
{
  // output
  std::ofstream outfile;
  outfile.open(filename);
  time_t curtime = time(0);
  tm now=*localtime(&curtime);
  char dest[BUFSIZ]= {0};
  strftime(dest, sizeof(dest)-1, "%c", &now);
  outfile << "# " << dest << std::endl << std::endl
  << "# input dimensionality" << std::endl << input_dim << std::endl
  << std::endl << "# covariance function" << std::endl
  << cf->to_string() << std::endl << std::endl
  << "# log-hyperparameter" << std::endl;
  Eigen::VectorXd param = cf->get_loghyper();
  for (size_t i = 0; i< cf->get_param_dim(); i++) {
    outfile << std::setprecision(10) << param(i) << " ";
  }
  outfile << std::endl << std::endl
  << "# data (target value in first column)" << std::endl;
  for (size_t i=0; i<sampleset->size(); ++i) {
    outfile << std::setprecision(10) << sampleset->y(i) << " ";
    for(size_t j = 0; j < input_dim; ++j) {
      outfile << std::setprecision(10) << sampleset->x(i)(j) << " ";
    }
    outfile << std::endl;
  }
  outfile.close();
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

void AbstractGaussianProcess::add_pattern(const Eigen::VectorXd & x, double y)
{
  //std::cout<< L.rows() << std::endl;
  sampleset->add(x, y);
  alpha_needs_update = true;
  updateCholesky(x.data(), y);
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

Eigen::VectorXd AbstractGaussianProcess::getAlpha(){
	compute();
	return alpha;
}

Eigen::MatrixXd AbstractGaussianProcess::getL(){
	compute();
	//make sure that the upper part is deleted
	return L.triangularView<Eigen::Lower>();
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

