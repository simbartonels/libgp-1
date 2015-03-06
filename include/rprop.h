// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __RPROP_H__
#define __RPROP_H__

#include "abstract_gp.h"
#include <Eigen/Core>

namespace libgp {

/** Gradient-based optimizer.
 *  @author Manuel Blum */
class RProp {
public:
	RProp() {
		init();
	}
	void init(double eps_stop = 0.0, double Delta0 = 0.1,
			double Deltamin = 1e-6, double Deltamax = 50, double etaminus = 0.5,
			double etaplus = 1.2);
	void maximize(AbstractGaussianProcess * gp, size_t n = 100,
			bool verbose = 1);

	/**
	 * Maximizes the log-likelihood of the Gaussian process and writes the sequence of parameters
	 * found in param_history. The size of times will be interpreted as number of rprop steps. Thus
	 * each of the next matrices and vectors MUST have these as number of columns.
	 * Param_history MUST have #parameters rows.
	 * Further this method will write the negative log-likelihood, test mean and test variance
	 * into the	respective matrices for the currently best parameter. Please make sure these
	 * matrices are of appropriate size: number of test inputs x iters.
	 */
	void maximize(AbstractGaussianProcess * gp, const Eigen::MatrixXd & testX,
			Eigen::VectorXd & times, Eigen::MatrixXd & param_history, Eigen::MatrixXd & meanY,
			Eigen::MatrixXd & varY, Eigen::VectorXd & nllh);
private:

	/**
	 * Performs an RProp step. Returns NAN if converged.
	 */
	inline double step(AbstractGaussianProcess * gp, double & best,
			Eigen::VectorXd & Delta, Eigen::VectorXd & grad_old,
			Eigen::VectorXd & params, Eigen::VectorXd & best_params);
	double Delta0;
	double Deltamin;
	double Deltamax;
	double etaminus;
	double etaplus;
	double eps_stop;
};
}

#endif /* __RPROP_H__ */
