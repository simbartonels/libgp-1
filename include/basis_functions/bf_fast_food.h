// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_

#include "IBasisFunction.h"

namespace libgp{
class FastFood : public IBasisFunction{
public:
		Eigen::VectorXd computeBasisFunctionVector(const Eigen::VectorXd &x);

		Eigen::MatrixXd getInverseWeightPrior();

		Eigen::MatrixXd getCholeskyOfInverseWeightPrior();

		Eigen::MatrixXd getWeightPrior();

	    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);

	    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, double kernel_value, Eigen::VectorXd &grad);

		void gradBasisFunction(const Eigen::VectorXd &x, const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad);

		void gradInverseWeightPrior(size_t p, Eigen::MatrixXd & diSigmadp);

	    std::string to_string();


	protected:
	    virtual bool real_init();
};
}


#endif /* INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_ */
