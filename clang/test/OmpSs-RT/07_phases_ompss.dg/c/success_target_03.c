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
// XFAIL: *


/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/

#include <stdlib.h>
#include <stdio.h>

#define n 20
const int M = n;

int b[n];

int main (int argc, char *argv[])
{
    int *a = b;

    {
        int k;
        for (k = 0; k < M; k++)
        {
            a[k] = k + 1;
        }
    }

#pragma oss target device(smp) copy_in(a[0:M-1])
#pragma oss task firstprivate(stderr) firstprivate(M)
    {
        int k;
        for (k = 0; k < M; k++)
        {
            if (a[k] != (k + 1))
            {
                fprintf(stderr, "a[%d] == %d != %d\n", k, a[k], k+1);
                abort();
            }
        }
    }

#pragma oss taskwait

    return 0;
}

