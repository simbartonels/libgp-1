/*
 * time_call.h
 *
 *  Created on: Jan 8, 2015
 *      Author: raven
 */

#ifndef BENCHMARKS_UTIL_TIME_CALL_H_
#define BENCHMARKS_UTIL_TIME_CALL_H_

#include <iostream>


void compare_time(void (*f1)(), void (*f2)(), size_t calls);

#endif /* BENCHMARKS_UTIL_TIME_CALL_H_ */
