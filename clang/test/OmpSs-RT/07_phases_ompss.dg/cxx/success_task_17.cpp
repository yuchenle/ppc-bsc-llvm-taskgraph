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

// RUN: %oss-cxx-compile-and-run
// RUN: %oss-cxx-O2-compile-and-run

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/

#include <cstdlib>
#include <map>

int main (int argc, char *argv[])
{
	std::map<int, int> m;

    for (int i = 0; i < 100; i++)
    {
        m[i] = i;
    }

	for (std::map<int, int>::iterator it = m.begin();
            it != m.end();
            it++)
	{
		#pragma oss task
		{
            it->second++;
		}
	}

#pragma oss taskwait

    for (int i = 0; i < 100; i++)
    {
        if (m[i] != (i + 1))
            abort();
    }

    return 0;
}
