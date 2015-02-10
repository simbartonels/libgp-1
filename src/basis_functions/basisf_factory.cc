// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <string>
#include <cassert>

#include "basis_functions/basisf_factory.h"
#include "basis_functions/bf_multi_scale.h"
#include "basis_functions/bf_solin.h"
//#ifdef INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_
#ifdef BUILD_FAST_FOOD
#	include "basis_functions/bf_fast_food.h"
#endif

namespace libgp {
  
BasisFFactory::BasisFFactory () {
	//TODO: these hard coded names ain't nice. use getName for that
	registry["SparseMultiScaleGP"] = & create_func<MultiScale>;
	registry["Solin"] = & create_func<Solin>;
	#ifdef BUILD_FAST_FOOD
	registry["FastFood"] = & create_func<FastFood>;
	#endif
  }
  
  BasisFFactory::~BasisFFactory () {};
  
  IBasisFunction* BasisFFactory::createBasisFunction(const std::string key, size_t num_basisf, CovarianceFunction * wrapped_cov_func) {
    IBasisFunction * bf = findBasisFunction(key);
    bool initialized = bf->init(num_basisf, wrapped_cov_func);
    assert(initialized);
    return bf;
  }

  IBasisFunction* BasisFFactory::createBasisFunction(const std::string key, size_t num_basisf, CovarianceFunction * wrapped_cov_func, size_t seed) {
    IBasisFunction * bf = findBasisFunction(key);
	bool initialized = bf->init(num_basisf, wrapped_cov_func, seed);
    assert(initialized);
    return bf;
  }

  IBasisFunction* BasisFFactory::findBasisFunction(const std::string key){
	    IBasisFunction * bf = NULL;

	    std::map<std::string , BasisFFactory::create_func_def>::iterator it = registry.find(key);
	    if (it == registry.end()) {
	      std::cerr << "fatal error while parsing basis function: " << key << " not found" << std::endl;
	      exit(0);
	    }
	    bf = registry.find(key)->second();
	    return bf;
  }

  std::vector<std::string> BasisFFactory::list()
  {
    std::vector<std::string> products;
    std::map<std::string , BasisFFactory::create_func_def>::iterator it;
    for (it = registry.begin(); it != registry.end(); ++it) {
      products.push_back((*it).first);
    }
    return products;
  }
}
