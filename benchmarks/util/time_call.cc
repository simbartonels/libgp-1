#include "time_call.h"

#include <sys/time.h>
#include <cmath>

static timestamp_t mytime ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

void compare_time(void (*f1)(), void (*f2)(), size_t calls){
	timestamp_t fast = 0;
	timestamp_t naive = 0;
	timestamp_t t = 0;

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

static timestamp_t last_stop = mytime();

timestamp_t stop_watch(){
	timestamp_t now = mytime();
	timestamp_t ret = now - last_stop;
	last_stop = now;
	return ret;
}

