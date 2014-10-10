// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"

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

  //TODO: can this method also be moved to the abstract class?
//  GaussianProcess::GaussianProcess (const char * filename)
//  {
//    int stage = 0;
//    std::ifstream infile;
//    double y;
//    infile.open(filename);
//    std::string s;
//    double * x = NULL;
//    L.resize(initial_L_size, initial_L_size);
//    while (infile.good()) {
//      getline(infile, s);
//      // ignore empty lines and comments
//      if (s.length() != 0 && s.at(0) != '#') {
//        std::stringstream ss(s);
//        if (stage > 2) {
//          ss >> y;
//          for(size_t j = 0; j < input_dim; ++j) {
//            ss >> x[j];
//          }
//          add_pattern(x, y);
//        } else if (stage == 0) {
//          ss >> input_dim;
//          sampleset = new SampleSet(input_dim);
//          x = new double[input_dim];
//        } else if (stage == 1) {
//          CovFactory factory;
//          cf = factory.create(input_dim, s);
//          cf->loghyper_changed = 0;
//        } else if (stage == 2) {
//          Eigen::VectorXd params(cf->get_param_dim());
//          for (size_t j = 0; j<cf->get_param_dim(); ++j) {
//            ss >> params[j];
//          }
//          cf->set_loghyper(params);
//        }
//        stage++;
//      }
//    }
//    infile.close();
//    if (stage < 3) {
//      std::cerr << "fatal error while reading " << filename << std::endl;
//      exit(EXIT_FAILURE);
//    }
//    delete [] x;
//  }
  
  double GaussianProcess::var_impl(const Eigen::VectorXd x_star){
	  int n = sampleset->size();
	  Eigen::VectorXd v = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
	  return cf->get(x_star, x_star) - v.dot(v);
  }

  void GaussianProcess::computeCholesky()
  {
    int n = sampleset->size();
    // resize L if necessary
    if (n > L.rows()) L.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        L(i, j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L.topLeftCorner(n, n) = L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
  }
  
  void GaussianProcess::updateCholesky(const double x[], double y){
	  int n = sampleset->size() - 1;

	  // create kernel matrix if sampleset is empty
	  if (n == 0) {
	    L(0,0) = sqrt(cf->get(sampleset->x(0), sampleset->x(0)));
	    cf->loghyper_changed = false;
	  // recompute kernel matrix if necessary
	  } else if (cf->loghyper_changed) {
	    computeCholesky();
	    cf->loghyper_changed = false;
	  // update kernel matrix
	  } else {
	    Eigen::VectorXd k(n);

	    for (int i = 0; i<n; ++i) {
	      k(i) = cf->get(sampleset->x(i), sampleset->x(n));
	    }

	    double kappa = cf->get(sampleset->x(n), sampleset->x(n));

	    // resize L if necessary
	    if (sampleset->size() > L.rows()) {
	      L.conservativeResize(n + initial_L_size, n + initial_L_size);
	    }
	    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
	    L.block(n,0,1,n) = k.transpose();
	    L(n,n) = sqrt(kappa - k.dot(k));
	  }
  }

  void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    k_star.resize(sampleset->size());
    for(size_t i = 0; i < sampleset->size(); ++i) {
      k_star(i) = cf->get(x_star, sampleset->x(i));
    }
  }

  void GaussianProcess::update_alpha()
  {
    alpha.resize(sampleset->size());
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    int n = sampleset->size();
    alpha = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha);
  }
  
  void GaussianProcess::write(const char * filename)
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

  double GaussianProcess::log_likelihood_impl()
  {
    int n = sampleset->size();
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    double det = 2 * L.diagonal().head(n).array().log().sum();
    return -0.5*y.dot(alpha) - 0.5*det - 0.5*n*log2pi;
  }

  Eigen::VectorXd GaussianProcess::log_likelihood_gradient_impl()
  {
    size_t n = sampleset->size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);

    W = alpha * alpha.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset->x(i), sampleset->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
  }
}
