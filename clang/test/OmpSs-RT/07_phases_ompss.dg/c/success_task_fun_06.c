/*--------------------------------------------------------------------
  (C) Copyright 2006-2012 Barcelona Supercomputing Center
                          Centro Nacional de Supercomputacion
  
  This file is part of Mercurium C/C++ source-to-source compiler.
  
  See AUTHORS file in the top level directory for information
  regarding developers and contributors.
  
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.
  
  Mercurium C/C++ source-to-source compiler is distributed in the hope
  that it will be useful, but WITHOUT ANY WARRANTY; without even the
  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the GNU Lesser General Public License for more
  details.
  
  You should have received a copy of the GNU Lesser General Public
  License along with Mercurium C/C++ source-to-source compiler; if
  not, write to the Free Software Foundation, Inc., 675 Mass Ave,
  Cambridge, MA 02139, USA.
--------------------------------------------------------------------*/

// RUN: %oss-compile-and-run
// RUN: %oss-O2-compile-and-run

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
test_CFLAGS="--no-copy-deps"
</testinfo>
*/
#include <assert.h>

#pragma oss task  inout([1] var)
void f(int * var, int cnst)
{
    int i = 0;
    double x = 1;
    for (i = 0; i < 10000; ++i)
    {
        x = x * 2.0;
    }
    assert(*var == cnst);
    (*var) = (*var) + 1;
}

#pragma oss task  concurrent([1] var)
void g(int * var)
{
    (*var) = 0;
}

int main()
{
    int i;
    int result = 0;
    int *ptrResult = &result;
    for (i = 0; i < 10; i++)
        f(ptrResult, i);

    g(ptrResult);

    for (i = 0; i < 10; i++)
        f(ptrResult, i);
#pragma oss taskwait

}

