/*
 * circuit.cpp - circuit class implementation
 *
 * Copyright (C) 2003, 2004 Stefan Jahn <stefan@lkcc.org>
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this package; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.  
 *
 * $Id: circuit.cpp,v 1.6 2004/01/30 21:40:35 ela Exp $
 *
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logging.h"
#include "complex.h"
#include "object.h"
#include "node.h"
#include "property.h"
#include "circuit.h"
#include "substrate.h"
#include "component_id.h"

// Constructor creates an unnamed instance of the circuit class.
circuit::circuit () : object () {
  size = 0;
  data = NULL;
  nodes = NULL;
  port = 0;
  org = 1;
  subst = NULL;
  source = 0;
  nSources = 0;
  type = CIR_UNKNOWN;
}

/* Constructor creates an unnamed instance of the circuit class with a
   certain number of ports. */
circuit::circuit (int s) : object () {
  size = s;
  data = new complex[s * s];
  nodes = new node[s];
  port = 0;
  org = 1;
  subst = NULL;
  source = 0;
  nSources = 0;
  type = CIR_UNKNOWN;
}

/* The copy constructor creates a new instance based on the given
   circuit object. */
circuit::circuit (const circuit & c) : object (c) {
  size = c.size;
  port = c.port;
  org = c.org;
  type = c.type;
  subst = c.subst;
  Y = c.Y;
  source = c.source;
  nSources = c.nSources;
  nodes = new node[size];
  data = new complex[size * size];

  // copy each node and set its circuit to the current circuit object
  for (int i = 0; i < size; i++) {
    node * n = new node (c.nodes[i]);
    n->setCircuit (this);
    nodes[i] = * n;
    delete n;
  }
  // copy each s parameter
  for (int i = 0; i < size * size; i++) {
    complex * z = new complex (c.data[i]);
    data[i] = * z;
    delete z;
  }
}

// Destructor deletes a circuit object.
circuit::~circuit () {
  delete[] data;
  delete[] nodes;
}

/* This function sets the name and port number of one of the circuit's
   nodes. */
void circuit::setNode (int i, char * n) {
  nodes[i - 1].setName (n);
  nodes[i - 1].setCircuit (this);
  nodes[i - 1].setPort (i);
}

// Returns one of the circuit's nodes.
node * circuit::getNode (int i) {
  return &nodes[i - 1];
}

/* This function counts the number of signals (ports) within the list
   of circuits. */
int circuit::countPorts (void) {
  int count = 0;
  for (circuit * c = this; c != NULL; c = (circuit *) c->getNext ()) {
    if (c->isPort ())
      count++;
  }
  return count;
}

/* This function counts the number of circuits within the list 
   of circuits. */
int circuit::countCircuits (void) {
  int count = 0;
  for (circuit * c = this; c != NULL; c = (circuit *) c->getNext ()) {
    count++;
  }
  return count;
}

/* This function counts the number of circuits within the list 
   of circuits. */
int circuit::countNodes (void) {
  int count = 0;
  for (circuit * c = this; c != NULL; c = (circuit *) c->getNext ()) {
    if (!c->isPort ())
      count += c->getSize ();
  }
  return count;
}

#if DEBUG
// DEBUG function:  Prints the S parameters of this circuit object.
void circuit::print (void) {
  for (int i = 1; i <= getSize (); i++) {
    for (int j = 1; j <= getSize (); j++) {
      logprint (LOG_STATUS, "%s S%d%d(%+.3e,%+.3e) ", getName (), i, j, 
		(double) real (getS (i, j)), (double) imag (getS (i, j)));
    }
    logprint (LOG_STATUS, "\n");
  }
}
#endif /* DEBUG */

/* Returns the current substrate of the circuit object.  Used for
   microstrip components only. */
substrate * circuit::getSubstrate (void) { 
  return subst;
}

// Sets the substrate of the circuit object.
void circuit::setSubstrate (substrate * s) {
  subst = s;
}

/* Returns the circuits B-MNA matrix value of the given voltage source
   built in the circuit depending on the port number. */
complex circuit::getB (int nr, int port) {
  return MatrixB[(nr - source) * size + port - 1];
}

/* Sets the circuits B-MNA matrix value of the given voltage source
   built in the circuit depending on the port number. */
void circuit::setB (int nr, int port, complex z) {
  MatrixB[(nr - 1) * size + port - 1] = z;
}

/* Returns the circuits C-MNA matrix value of the given voltage source
   built in the circuit depending on the port number. */
complex circuit::getC (int nr, int port) {
  return MatrixC[(nr - source) * size + port - 1];
}

/* Sets the circuits C-MNA matrix value of the given voltage source
   built in the circuit depending on the port number. */
void circuit::setC (int nr, int port, complex z) {
  MatrixC[(nr - 1) * size + port - 1] = z;
}

/* Returns the circuits D-MNA matrix value of the given voltage source
   built in the circuit. */
complex circuit::getD (int nr) {
  return MatrixD[nr - source];
}

/* Sets the circuits D-MNA matrix value of the given voltage source
   built in the circuit. */
void circuit::setD (int nr, complex z) {
  MatrixD[nr - 1] = z;
}

/* Returns the circuits E-MNA matrix value of the given voltage source
   built in the circuit. */
complex circuit::getE (int nr) {
  return MatrixE[nr - source];
}

/* Sets the circuits E-MNA matrix value of the given voltage source
   built in the circuit. */
void circuit::setE (int nr, complex z) {
  MatrixE[nr - 1] = z;
}

/* Returns the circuits I-MNA matrix value of the current source built
   in the circuit. */
complex circuit::getI (int port) {
  return MatrixI[port - 1];
}

/* Sets the circuits I-MNA matrix value of the current source built in
   the circuit depending on the port number. */
void circuit::setI (int port, complex z) {
  MatrixI[port - 1] = z;
}