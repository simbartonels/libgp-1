// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __BASISF_FACTORY_H__
#define __BASISF_FACTORY_H__

#include <iostream>
#include <sstream>
#include <vector>
#include <map>

#include "IBasisFunction.h"
#include "cov.h"

namespace libgp {

template<typename ClassName> IBasisFunction * create_func() {
	return new ClassName();
}

/** Factory class for generating instances of IBasisFunction.
 *  @author Manuel Blum, Simon Bartels */
class BasisFFactory {
public:

	BasisFFactory();
	virtual ~BasisFFactory();

	/**
	 *  Creates an instance of CovarianceFunction that implements IBasisFunction.
	 *  @param key string representation of a basis function
	 *  @param M the number of basis functions
	 *  @param wrapped_cov_func the wrapped covariance function
	 *  @return instance of IBasisFunction
	 */
	libgp::IBasisFunction* createBasisFunction(const std::string key, size_t M, CovarianceFunction * wrapped_cov_func);

	/** Returns a string vector of available covariance functions. */
	std::vector<std::string> list();

private:
	typedef IBasisFunction*(*create_func_def)();
	std::map<std::string, BasisFFactory::create_func_def> registry;
};
}

#endif /* __BASISF_FACTORY_H__ */
