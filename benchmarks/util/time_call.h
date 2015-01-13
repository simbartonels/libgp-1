/*
 * time_call.h
 *
 *  Created on: Jan 8, 2015
 *      Author: raven
 */

#ifndef BENCHMARKS_UTIL_TIME_CALL_H_
#define BENCHMARKS_UTIL_TIME_CALL_H_

#include <iostream>

typedef unsigned long long timestamp_t;

void compare_time(void (*f1)(), void (*f2)(), size_t calls);

timestamp_t stop_watch();

#endif /* BENCHMARKS_UTIL_TIME_CALL_H_ */
