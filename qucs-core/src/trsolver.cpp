/*
 * trsolver.cpp - transient solver class implementation
 *
 * Copyright (C) 2004 Stefan Jahn <stefan@lkcc.org>
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
 * $Id: trsolver.cpp,v 1.6 2004/09/14 19:33:09 ela Exp $
 *
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "object.h"
#include "logging.h"
#include "complex.h"
#include "circuit.h"
#include "sweep.h"
#include "net.h"
#include "analysis.h"
#include "nasolver.h"
#include "trsolver.h"
#include "transient.h"

#define dState 0 // delta T state

// Constructor creates an unnamed instance of the trsolver class.
trsolver::trsolver ()
  : nasolver<nr_double_t> (), states<nr_double_t> () {
  swp = NULL;
  type = ANALYSIS_TRANSIENT;
  setDescription ("transient");
}

// Constructor creates a named instance of the trsolver class.
trsolver::trsolver (char * n)
  : nasolver<nr_double_t> (n), states<nr_double_t> () {
  swp = NULL;
  type = ANALYSIS_TRANSIENT;
  setDescription ("transient");
}

// Destructor deletes the trsolver class object.
trsolver::~trsolver () {
  if (swp) delete swp;
}

/* The copy constructor creates a new instance of the trsolver class
   based on the given trsolver object. */
trsolver::trsolver (trsolver & o)
  : nasolver<nr_double_t> (o), states<nr_double_t> (o) {
  swp = o.swp ? new sweep (*o.swp) : NULL;
}

// This function creates the time sweep if necessary.
void trsolver::initSteps (void) {
  if (swp != NULL) return;

  char * type = getPropertyString ("Type");
  nr_double_t start = getPropertyDouble ("Start");
  nr_double_t stop = getPropertyDouble ("Stop");
  int points = getPropertyInteger ("Points");

  if (!strcmp (type, "lin")) {
    swp = new linsweep ("time");
    ((linsweep *) swp)->create (start, stop, points);
  }
  else if (!strcmp (type, "log")) {
    swp = new logsweep ("time");
    ((logsweep *) swp)->create (start, stop, points);
  }
}

/* This is the transient netlist solver.  It prepares the circuit list
   for each requested time and solves it then. */
void trsolver::solve (void) {
  nr_double_t time, current = 0;
  int error = 0;
  runs++;

  // Create time sweep if necessary.
  initSteps ();

  // First calculate a initial state using the non-linear DC analysis.
  setDescription ("DC");
  initDC ();
  solve_pre ();
  error = solve_nonlinear ();
  if (error) {
    logprint (LOG_ERROR, "ERROR: %: initial DC analysis failed\n", getName ());
    return;
  }

  // Initialize transient analysis.
  setDescription ("transient");
  init ();
  swp->reset ();

  int running = 0;
  //delta /= 10;
  //updateCoefficients (delta, 1);
  //updateCoefficients (delta, 1);

  // Start to sweep through time.
  for (int i = 0; i < swp->getSize (); i++) {
    time = swp->next ();
    logprogressbar (i, swp->getSize (), 40);

#if DEBUG && 0
    logprint (LOG_STATUS, "NOTIFY: %s: solving netlist for t = %e\n",
	      getName (), (double) time);
#endif

    do {
      calc (current);
      error += solve_linear (); // Start the linear solver.

      nr_double_t save = delta;
      if (running > 2) {
	delta = checkDelta ();
	if (delta > deltaMax) delta = deltaMax;
	if (delta < deltaMin) delta = deltaMin;

	if (delta > 0.9 * save) {
	  updateCoefficients (delta, 1);
	  rejected = 0;
#if DEBUG && 0
	  logprint (LOG_STATUS,
		    "DEBUG: delta accepted at t = %.3e, h = %.3e\n",
		    current, delta);
#endif
	  savePreviousIteration ();
	}
	else if (save > delta) {
	  updateCoefficients (delta);
	  rejected++;
#if DEBUG && 0
	  logprint (LOG_STATUS,
		    "DEBUG: delta rejected at t = %.3e, h = %.3e\n",
		    current, delta);
#endif
	  current -= delta;
	} else {
	  updateCoefficients (delta, 1);
	}
      }
      else {
	updateCoefficients (delta, 1);
      }
      current += delta;
      running++;
    }
    while (current < time);
    
    // Save results.
#if DEBUG && 0
    logprint (LOG_STATUS, "DEBUG: save point at t = %.3e, h = %.3e\n",
	      current, delta);
#endif
    saveAllResults (time);
  }
  solve_post ();
  logprogressclear (40);
}

/* Goes through the list of circuit objects and runs its calcDC()
   function. */
void trsolver::calc (void) {
  circuit * root = subnet->getRoot ();
  for (circuit * c = root; c != NULL; c = (circuit *) c->getNext ()) {
    c->calcDC ();
  }
}

/* Goes through the list of circuit objects and runs its calcTR()
   function. */
void trsolver::calc (nr_double_t time) {
  circuit * root = subnet->getRoot ();
  for (circuit * c = root; c != NULL; c = (circuit *) c->getNext ()) {
    if (!rejected) c->nextState ();
    c->calcTR (time);
  }
}

/* Goes through the list of circuit objects and runs its initDC()
   function. */
void trsolver::initDC (void) {
  circuit * root = subnet->getRoot ();
  for (circuit * c = root; c != NULL; c = (circuit *) c->getNext ()) {
    c->initDC ();
  }
}

/* Goes through the list of circuit objects and runs its initTR()
   function. */
void trsolver::init (void) {
  IMethod = getPropertyString ("IntegrationMethod");
  order = getPropertyInteger ("Order");
  nr_double_t start = getPropertyDouble ("Start");
  nr_double_t stop = getPropertyDouble ("Stop");

  // initialize step values
  delta = getPropertyDouble ("InitialStep");
  deltaMin = getPropertyDouble ("MinStep");
  deltaMax = getPropertyDouble ("MaxStep");
  if (deltaMax == 0.0) deltaMax = (stop - start) / 50;
  if (deltaMin == 0.0) deltaMin = 1e-11 * deltaMax;
  if (delta == 0.0) delta = MIN (stop / 200, deltaMax) / 10;
  if (delta < deltaMin) delta = deltaMin;
  if (delta > deltaMax) delta = deltaMax;
  
  calcCoefficients (IMethod, order, coefficients, delta);

  // initialize step history
  setStates (1);
  initStates ();
  fillState (dState, delta);

  circuit * root = subnet->getRoot ();
  for (circuit * c = root; c != NULL; c = (circuit *) c->getNext ()) {
    c->initTR ();
    c->initStates ();
    c->setCoefficients (coefficients);
    c->setOrder (order);
    setIntegrationMethod (c, IMethod);
  }
}

/* This function saves the results of a single solve() functionality
   (for the given timestamp) into the output dataset. */
void trsolver::saveAllResults (nr_double_t time) {
  vector * t;
  // add current frequency to the dependency of the output dataset
  if ((t = data->findDependency ("time")) == NULL) {
    t = new vector ("time");
    data->addDependency (t);
  }
  if (runs == 1) t->add (time);
  saveResults ("Vt", "It", 0, t);
}

/* This function is meant to adapt the current timestep the transient
   analysis advanced.  For the computation of the new timestep the
   truncation error depending on the integration method is used. */
nr_double_t trsolver::checkDelta (void) {
  nr_double_t reltol = getPropertyDouble ("reltolTR");
  nr_double_t abstol = getPropertyDouble ("abstolTR");
  nr_double_t dif, rel, tol, n = DBL_MAX;
  int N = countNodes ();

  reltol = 1e-3;
  abstol = 1e-6;

  for (int r = 1; r <= N; r++) {
    dif = z->get (r, 1) - zprev->get (r, 1);
    rel = abs (x->get (r, 1));
    tol = reltol * rel + abstol;

    if (!strcmp (IMethod, "Euler")) {
      if (dif != 0) {
	nr_double_t t = delta * tol * 2 / dif;
	t = sqrt (fabs (t));
	n = MIN (n, t);
      }
    }
    else if (!strcmp (IMethod, "Trapezoidal")) {
      if (dif != 0) {
	nr_double_t del = delta + getState (dState, 1);
	nr_double_t t = tol * 12 * del / dif;
	t = exp (log (fabs (t)) / 3);
	n = MIN (n, t);
      }
    }
    else if (!strcmp (IMethod, "Gear")) {
      nr_double_t del = 0;
      for (int i = 0; i <= order; i++) del += getState (dState, i);
      if (dif != 0) {
	nr_double_t t = tol * del / dif;
	t = exp (log (fabs (t)) / (order + 1));
	n = MIN (n, t);
      }
    }
  }
  delta = MIN (2 * delta, n);
  return delta;
}

// The function updates the integration coefficients.
void trsolver::updateCoefficients (nr_double_t delta, int next) {
  nr_double_t c = getState (dState);
  coefficients[0] *= c / delta;
  if (next) {
    for (int i = 1; i < 8; i++) {
      //coefficients[i] *= getState (dState, i) / getState (dState, i - 1);
      coefficients[i] *= c / delta;
    }
    nextState ();
  }
  setState (dState, delta);
}