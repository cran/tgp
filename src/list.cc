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

extern "C"
{
#include "rhelp.h"
}
#include "list.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>


/*
 * the usual constructor function 
 * for NODE
 */

LNode::LNode(void* entry)
{
	this->entry = entry;
	prev = next = NULL;
	list = NULL;
}


/*
 * the usual destructor function
 * for NODE
 */

LNode::~LNode(void)
{
}


/*
 * return the next node in the list
 */

LNode* LNode::Next(void) 
{
	return next;	
}


/*
 * return the previous node in the list
 */

LNode* LNode::Prev(void) 
{
	return prev;	
}


/*
 * return the data entry for the node
 */

void* LNode::Entry(void)
{
	return entry;
}

/*
 ************************
 * BEGIN List FUNCTIONS *
 ************************
 */


/*
 * the usual constructor function 
 * for LIST
 */

List::List(void)
{
	first = last = curr = NULL;
	len = 0;
}


/*
 * the usual destructor function
 * for LIST
 */

List::~List(void)
{
	curr = first;
	if(curr) warning("nonempty list deleted");
	while(curr) {
		LNode* temp = curr;
		curr = curr->next;
		delete temp;
	}
}


/*
 * insert a new node at the beginning
 * of the list
 */

LNode* List::EnQueue(void* entry)
{
	if(first == NULL) {
		assert(last == NULL);
		assert(len == 0);
		first = new LNode(entry);
		last = first;
	} else {
		assert(last != NULL);
		assert(len > 0);
		LNode* newnode = new LNode(entry);
		newnode->next = first;
		assert(first->prev == NULL);
		first->prev = newnode;
		first = newnode;
	}
	len++;
	first->list = this;
	return first;
}


/*
 * remove a node from the end 
 * of the list
 */

void * List::DeQueue(void)
{
	if(last == NULL) {
		assert(first == NULL);
		assert(len == 0);
		return NULL;
	} else {
		LNode* temp = last;
		if(first == last) {
			first = NULL;
		} else {
			assert(last->prev != NULL);
			last->prev->next = NULL;
		}
		last = last->prev;
		len--;
		assert(len >= 0);

		void* entry = temp->Entry();
		temp->list = NULL;
		delete temp;
		return entry;
	}
}


/*
 * check if the list is empty
 */

bool List::isEmpty(void)
{
	if(first == NULL) {
		assert(last == NULL);
		assert(len == 0);
		return true;
	} else {
		assert(last != NULL);
		assert(len > 0);
		return false;
	}
}


/*
 * return the length of the list
 */

unsigned int List::Len(void)
{
	return len;
}


/*
 * detach and delete the node from the list
 */

void* List::detach_and_delete(LNode* node)
{
	assert(node);
	if(node->list == NULL) {
		void* entry = node->Entry();
		delete node;
		return entry;
	}

	assert(node->list == this);
	if(node == first) {
		assert(node->prev == NULL);
		if(node == last) { /* first and last (one node list) */
			assert(node->next == NULL);
			first = last = NULL;
		} else { /* first but not last */
			assert(node->next != NULL);
			first = node->next;
			node->next = NULL;
			first->prev = NULL;
		}
	} else if(node == last) { /* last but not first */
		assert(node->next == NULL);
		assert(node->prev != NULL);
		last = node->prev;
		node->prev = NULL;
		last->next = NULL;
	} else { /* not last or first */
		node->prev->next = node->next;
		node->next->prev = node->prev;
		node->next = NULL;
		node->prev = NULL;
	}
	void* entry = node->Entry();
	node->list = NULL;
	delete node;
	node = NULL;
	len--;
	assert(len >= 0);
	return entry;
}


/*
 * return the first node in the list
 */

LNode* List::First(void)
{
	return first;
}
