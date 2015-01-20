#include <cmath>

#include "naive/bf_multi_scale_naive.h"
#include "basis_functions/bf_multi_scale.h"
#include "naive/bf_fast_food_naive.h"
#include "basis_functions/bf_fast_food.h"
#include "gp_utils.h"
#include "util/time_call.h"
#include "cov_factory.h"
#include "cov.h"
#include "sampleset.h"


using namespace libgp;

	static IBasisFunction * impl;
	static IBasisFunction * naive;

	static Eigen::VectorXd x;

	void f1bf(){
		impl->computeBasisFunctionVector(x);
	}

	void f2bf(){
		naive->computeBasisFunctionVector(x);
	}

	void testSpeedOfBasisFunction(){
		Eigen::VectorXd diff = impl->computeBasisFunctionVector(x) - naive->computeBasisFunctionVector(x);
		diff.array() = diff.array().abs();
		diff.array() /= (impl->computeBasisFunctionVector(x).array().abs() + 1e-30);

		assert(diff.array().abs().maxCoeff() < 1e-5);

		compare_time(f1bf, f2bf, 20);
	}

	static SampleSet * sampleSet;
	static Eigen::MatrixXd Phi;
	static Eigen::MatrixXd Grad;
	static Eigen::VectorXd temp;
	static size_t p;


	void f1gradbf(){
		impl->gradBasisFunction(sampleSet, Phi, p, Grad);
	}

	void f2gradbf(){
		size_t n = sampleSet->size();
		for(size_t i = 0; i<n;i++){
			naive->gradBasisFunction(sampleSet->x(i), Phi.col(i), p, temp);
			Grad.col(i) = temp;
		}
	}

	void testSpeedOfGradBasisFunction(){
    	// add training patterns
		size_t n = 2000;
		size_t M = impl->getNumberOfBasisFunctions();
		size_t D = x.size();

		Phi.resize(M, n);
		Grad.resize(M, n);
		temp.resize(M);
		sampleSet = new SampleSet(D);
    	for (int i = 0; i < n; ++i) {
    		double x[] = { drand48() * 4 - 2, drand48() * 4 - 2 };
    		double y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
    		sampleSet->add(x, y);
    		Phi.col(i) = impl->computeBasisFunctionVector(sampleSet->x(i));
    	}

    	size_t param_dim = impl->get_param_dim();
    	for(size_t i = 0; i < param_dim; i++){
//    		p = i*M*D+D-1;
    		p = i;
    		compare_time(f1gradbf, f2gradbf, 10);
    	}
	}

	void initFastFood(){
		((FastFood *)impl)->setPIs(((FastFoodNaive *) naive)->getPIs());
		((FastFood *)impl)->setB(((FastFoodNaive *) naive)->getB());
		((FastFood *)impl)->setG(((FastFoodNaive *) naive)->getG());
		((FastFood *)impl)->setS(((FastFoodNaive *) naive)->getS());
	}

	int main(int argc, char const *argv[]) {
		size_t D = 64;
		size_t M = 600;

		CovFactory f;
		CovarianceFunction * cov;
		cov = f.create(D, "CovSum ( CovSEard, CovNoise)");
		x.resize(D);
		x.setRandom();

//		impl = new MultiScale();
//		naive = new MultiScaleNaive();
		naive = new FastFoodNaive();
		impl = new FastFood();
		impl->init(M, cov);

		naive->init(M, cov);

		Eigen::VectorXd p(impl->get_param_dim());
		p.setRandom();
		impl->set_loghyper(p);
		naive->set_loghyper(p);

		initFastFood();

//		testSpeedOfGradBasisFunction();
		testSpeedOfBasisFunction();

		delete impl;
		delete naive;
		return 0;
	}

