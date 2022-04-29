/******************************************************************************
 * (C) Copyright 2020 AMIQ Consulting
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * NAME:        buffer.h
 * PROJECT:     conv
 * Description: Used to buffer first elements from an image
 *******************************************************************************/
 
#ifndef _BUFFER_H_
#define _BUFFER_H_
#include <cstdlib>

template<class T, int SIZE>
class buffer
{
public:
	buffer();
	void SetValue(T val, int pos);
	void InsertFront(T val);
	void InsertBack(T val);
	T GetValue(int pos);
private:
	T array[SIZE];
};


/////////////////////////////////////////////////
// buffer.cpp
/////////////////////////////////////////////////

template<class T, int SIZE>
buffer<T,SIZE>::buffer()
{
	memset(array, 0, SIZE*sizeof(T));
}


template<class T, int SIZE>
void buffer<T,SIZE>::SetValue(T val, int pos)
{
	array[pos] = val;
}

template<class T, int SIZE>
void buffer<T,SIZE>::InsertFront(T val)
{
	memmove(array+1,array,(SIZE-1)*sizeof(T));
	array[0] = val;
}


template<class T, int SIZE>
void buffer<T,SIZE>::InsertBack(T val)
{
	memmove(array,array+1,(SIZE-1)*sizeof(T));
	array[SIZE - 1] = val;
}


template<class T, int SIZE>
T buffer<T,SIZE>::GetValue(int pos)
{
	return array[pos];
}

#endif
