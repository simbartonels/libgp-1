#include "time_call.h"

#include <sys/time.h>
#include <cmath>

static double mytime ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec / 1000.0 + now.tv_sec * 1000.0;
}

void compare_time(void (*f1)(), void (*f2)(), size_t calls){
	double fast = 0;
	double naive = 0;
	double t = 0;

	t = mytime();
	(*f1)();
	t = mytime() - t;
	fast = t;

	t = mytime();
	(*f2)();
	t = mytime() - t;
	naive = t;
	for (size_t i = 0; i < calls; i++) {
		t = mytime();
		(*f1)();
		t = mytime() - t;
		if (t < fast)
			fast = t;

		t = mytime();
		(*f2)();
		t = mytime() - t;
		if (t < naive)
			naive = t;
	}
	std::cout << "f1: " << fast << std::endl;
	std::cout << "f2: " << naive << std::endl;
}

static double last_stop = mytime();

double stop_watch(){
	double now = mytime();
	double ret = now - last_stop;
	last_stop = now;
	return ret;
}

