/******************************************************************************** 
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C) 2005, University of California
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (rbgramacy@ams.ucsc.edu)
 *
 ********************************************************************************/


#ifndef __LIST_H__
#define __LIST_H__ 


class List;

class LNode
{
  private:
  	void* entry;

  public:

	List* list;

	LNode* next;
	LNode* prev;

	LNode(void* entry);
	~LNode(void);
	LNode* Next(void);
	LNode* Prev(void);
	void* Entry(void);
};


class List
{
  private:

  	LNode* first;
	LNode* last;
	LNode* curr;
	unsigned int len;

  public:
	List(void);
	~List(void);
	LNode* EnQueue(void *entry);
	void* DeQueue(void);
	bool isEmpty(void);
	unsigned int Len(void);
	void* detach_and_delete(LNode* node);
	LNode* First(void);
};


#endif
