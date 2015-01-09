//#include <omp.h>
#include <sys/time.h>
#include <cmath>

#include "gp_fic.h"
#include "gp_fic_naive.h"
#include "gp_utils.h"

using namespace libgp;

typedef unsigned long long timestamp_t;

    static timestamp_t
    mytime ()
    {
      struct timeval now;
      gettimeofday (&now, NULL);
      return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
    }

    static Eigen::VectorXd static_test;

    void testing(const Eigen::VectorXd& v){
    	std::cout << "v data: " << v.data() << std::endl;
    }

    void testing2(Eigen::VectorXd& out){
    	out(0) = -1.;
    }

    void testing3(Eigen::MatrixXd& out){
    	out(0, 0) = -2.;
    	std::cout << "outMat data: " << out.data() << std::endl;
    	std::cout << "outMat ref: " << &out << std::endl;
    }

    void testPassingReferences(){
    	size_t input_dim = 3;
    	size_t M = 200;
    	size_t n = 1000;

    	//TODO: find a way to extract a vector from a matrix without copying
    	//TODO: then adapt SampleSet

    	Eigen::MatrixXd Mat(M, M);
    	Mat.setRandom();
    	Eigen::Map<const Eigen::VectorXd> vec(Mat.row(0).data(), M);
    //	std::cout << "vec " << vec.transpose() << std::endl;
    //	std::cout << "M.col(0) " << Mat.col(0).transpose() << std::endl;

    	Eigen::VectorXd copyvec = vec;
    	std::cout << "copyvec data: " << copyvec.data() << std::endl;

    	//TODO: calls like these trigger a copy!
    	std::cout << "Mat(0) data: " << Mat.row(0).data() << std::endl;
    	testing(Mat.row(0));

    	std::cout << "Mat(0) data: " << Mat.col(0).data() << std::endl;
    	testing(Mat.col(0));

    	std::cout << "vec data: " << vec.data() << std::endl;
    	testing(vec);

    	std::cout << "before: " << copyvec(0) << std::endl;
    	testing2(copyvec);
    	std::cout << "after: " << copyvec(0) << std::endl;
    	std::cout << "M after: " << Mat(0, 0) << std::endl;

    	Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> MatMap(Mat.data(), M, M);
    	std::cout << "before: " << MatMap(0, 0) << std::endl;
    	std::cout << "Mat ref: " << &Mat << std::endl;
    	testing3(Mat);
    	std::cout << "after: " << MatMap(0, 0) << std::endl;
    	std::cout << "M after: " << Mat(0, 0) << std::endl;


    //	std::cout << "before: " << vec(0) << std::endl;
    //	testing2(vec);
    //	std::cout << "after: " << vec(0) << std::endl;
    //
    //	std::cout << "before: " << Mat(0, 0) << std::endl;
    //	testing2(Mat.col(0));
    //	std::cout << "after: " << Mat(0, 0) << std::endl;

//    	std::cout << "M before: " << Mat(0, 0) << std::endl;
//    	testing3(Mat.col(0));
//    	std::cout << "M after: " << Mat(0, 0) << std::endl;

    }



    void testSpeedOfSymmetricMatrixMultiplication(){
    	//TODO: implement
    	//1) A = Phi*PhiT
    	//2) for loop
    }

    Eigen::VectorXd returnVector(){
    	return static_test;
    }

    Eigen::VectorXd& returnVectorRef(){
    	return static_test;
    }

    const Eigen::VectorXd returnConstVector(){
    	return static_test;
    }

    const Eigen::VectorXd& returnConstVectorRef(){
    	return static_test;
    }

    void testCopying(){
    	static_test.resize(10);
    	static_test.setRandom();
    	//not allowed
//    	std::cout << "func ref: " << &(returnVector()) << std::endl;
    	Eigen::Map<Eigen::VectorXd> p(returnVector().data(), 10);
//    	std::cout << "func ref: " << p.data() << std::endl;
    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
    	p(0) = 0.;
    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

    	p = Eigen::Map<Eigen::VectorXd>(returnVectorRef().data(), 10);
    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
    	p(0) = 0.;
    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

//    	p = Eigen::Map<Eigen::VectorXd>(returnConstVector().data(), 10);
//    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
//    	p(0) = 0.;
//    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

    	std::cout << "vec ref: " << static_test.data() << std::endl;
    	std::cout << "func ref: " << returnConstVector().data() << std::endl;

//    	p = Eigen::Map<Eigen::VectorXd>(returnConstVectorRef().data(), 10);
//    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
//    	p(0) = 0.;
//    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

    	std::cout << "vec ref: " << static_test.data() << std::endl;
    	std::cout << "func ref: " << returnConstVectorRef().data() << std::endl;

    	const Eigen::VectorXd & constRef = returnConstVectorRef();
//    	double * ref = constRef.data();
//    	const Eigen::Map<Eigen::VectorXd> pconst(constRef.data(), 10);

    }

    void testSpeed(){
    	size_t input_dim = 3;
    	size_t M = 200;
    	size_t n = 1000;

    	FICGaussianProcess gp(input_dim, "CovSum ( CovSEard, CovNoise)", M,
    			"SparseMultiScaleGP");
    	// initialize hyper parameter vector
    	Eigen::VectorXd params(gp.covf().get_param_dim());
    	params.setRandom();
    	// set parameters of covariance function
    	gp.covf().set_loghyper(params);

    	FICnaiveGaussianProcess gpnaive(input_dim, "CovSum ( CovSEard, CovNoise)",
    			M, "SparseMultiScaleGP");
    	gpnaive.covf().set_loghyper(params);

    	// add training patterns
    	for (int i = 0; i < n; ++i) {
    		double x[] = { drand48() * 4 - 2, drand48() * 4 - 2 };
    		double y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
    		gp.add_pattern(x, y);
    		gpnaive.add_pattern(x, y);
    	}

    	double fv = gp.log_likelihood();
    	double nv = gpnaive.log_likelihood();
    //	std::cout << "fv: " << fv << std::endl;
    //	std::cout << "nv: " << nv << std::endl;
    	assert(std::abs(fv-nv)/nv <1e-5);

    	timestamp_t fast = 0;
    	timestamp_t naive = 0;
    	timestamp_t t = 0;

    	t = mytime();
    	gp.log_likelihood();
    	t = mytime() - t;
    	fast = t;

    	t = mytime();
    	gpnaive.log_likelihood();
    	t = mytime() - t;
    	naive = t;
    	for (int i = 0; i < 10; i++) {
    		t = mytime();
    		gp.log_likelihood();
    		t = mytime() - t;
    		if (t < fast)
    			fast = t;

    		t = mytime();
    		gpnaive.log_likelihood();
    		t = mytime() - t;
    		if (t < naive)
    			naive = t;
    	}
    	std::cout << "fast: " << fast << std::endl;
    	std::cout << "naive: " << naive << std::endl;

    }

int main(int argc, char const *argv[]) {
	testPassingReferences();
	return 0;
}
