// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <string>
#include <cassert>

#include "basis_functions/basisf_factory.h"
#include "basis_functions/bf_multi_scale.h"
#ifdef INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_
#include "basis_functions/bf_fast_food.h"
#endif

namespace libgp {
  
BasisFFactory::BasisFFactory () {
	registry["SparseMultiScaleGP"] = & create_func<MultiScale>;
	#ifdef INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_
	registry["FastFood"] = & create_func<FastFood>;
	#endif
  }
  
  BasisFFactory::~BasisFFactory () {};
  
  IBasisFunction* BasisFFactory::createBasisFunction(const std::string key, size_t num_basisf, CovarianceFunction * wrapped_cov_func) {
    IBasisFunction * bf = NULL;

    std::map<std::string , BasisFFactory::create_func_def>::iterator it = registry.find(key);
    if (it == registry.end()) {
      std::cerr << "fatal error while parsing basis function: " << key << " not found" << std::endl;
      exit(0);
    } 
    bf = registry.find(key)->second();
    bool initialized = bf->init(num_basisf, wrapped_cov_func);
    assert(initialized);
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
