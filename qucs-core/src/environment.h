/*
 * environment.h - variable environment class definitions
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
 * $Id: environment.h,v 1.1 2004/02/13 20:31:45 ela Exp $
 *
 */

#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

class variable;

class environment
{
 public:
  environment ();
  environment (char *);
  environment (const environment &);
  virtual ~environment ();
  void setName (char *);
  char * getName (void);
  void copyVariables (variable *);
  void deleteVariables (void);
  void addVariable (variable *);
  variable * getVariable (char *);

 private:
  char * name;
  variable * root;
};

#endif /* __ENVIRONMENT_H__ */