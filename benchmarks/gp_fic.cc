#include <cmath>

#include "gp_fic.h"
#include "naive/gp_fic_naive.h"
#include "gp_utils.h"
#include "util/time_call.h"

using namespace libgp;

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

    static FICGaussianProcess * gp;

    static FICnaiveGaussianProcess * gpnaive;

    void f1(){
    	gp->log_likelihood();
    }

    void f2(){
    	gpnaive->log_likelihood();
    }

    void testSpeedOfLogLikelihood(){
    	double fv = gp->log_likelihood();
    	double nv = gpnaive->log_likelihood();
    //	std::cout << "fv: " << fv << std::endl;
    //	std::cout << "nv: " << nv << std::endl;
    	assert(std::abs(fv-nv)/nv <1e-5);

    	std::cout << "speed of llh computation:" << std::endl;
    	compare_time(f1, f2, 5);
    }

    static Eigen::MatrixXd V;
    static Eigen::MatrixXd Sigma1;
    static Eigen::MatrixXd Sigma2;
    static Eigen::MatrixXd Sigma3;
    static Eigen::VectorXd dg;

    void f1CholComp(){
//    	for(size_t i = 0; i < 100; i++)
    	dg = V.array().square().colwise().sum().transpose();
    }

    void f2CholComp(){
//    	for(size_t i = 0; i < 100; i++)
    	dg = (V.transpose() * V).diagonal();
    }

    void f3CholComp(){
//    	Sigma1.setZero();
    	Sigma1.selfadjointView<Eigen::Lower>().rankUpdate(V);
    	dg = Sigma1.diagonal();
    }


    void testSpeedOfCholeskyComputation1(){
    	compare_time(f1CholComp, f2CholComp, 100);
    }

    void testSpeedOfCholeskyComputation11(){
    	compare_time(f1CholComp, f3CholComp, 100);
    }

    void f1CholComp2(){
    	gp->covf().loghyper_changed = true;
    	gp->getL();
    }

    void f2CholComp2(){
    	gpnaive->covf().loghyper_changed = true;
    	gpnaive->getL();
    }

    void testSpeedOfCholeskyComputation2(){
    	compare_time(f1CholComp2, f2CholComp2, 10);
    }

    void llhGradBaseline(){
    	gpnaive->log_likelihood_gradient();
    }

    void llhGradFast(){
    	gp->log_likelihood_gradient();
    }

    void LuuLu_base(){
    	Sigma3 = Sigma1 * Sigma2;
    }

    void LuuLu_2(){
    	Sigma3 = Sigma1.triangularView<Eigen::Lower>() * Sigma2;
    }

    void testSpeedOfLuuLu(){
        	compare_time(LuuLu_base, LuuLu_2, 10);
        }


int main(int argc, char const *argv[]) {
	size_t input_dim = 3;
	size_t M = 200;
	size_t n = 2000;
	V.resize(M, n);
	V.setRandom();
	Sigma1.resize(M, M);
	Sigma2.resize(M, M);
	Sigma3.resize(M, M);
	Sigma1.setRandom();
	Sigma2.setRandom();
	Sigma3.setRandom();
	gp = new FICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M,
			"SparseMultiScaleGP");

	// initialize hyper parameter vector
	Eigen::VectorXd params(gp->covf().get_param_dim());
	params.setRandom();
	// set parameters of covariance function
	gp->covf().set_loghyper(params);

	gpnaive = new FICnaiveGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)",
			M, "SparseMultiScaleGP");
	gpnaive->covf().set_loghyper(params);

	// add training patterns
	for (int i = 0; i < n; ++i) {
		double x[] = { drand48() * 4 - 2, drand48() * 4 - 2 };
		double y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
		gp->add_pattern(x, y);
		gpnaive->add_pattern(x, y);
	}

//	testSpeedOfLogLikelihood();
//	testSpeedOfCholeskyComputation2();

//	compare_time(llhGradBaseline, llhGradFast, 1);

//	testSpeedOfCholeskyComputation11();
	testSpeedOfLuuLu();

	delete gp;
	delete gpnaive;
	return 0;
}
