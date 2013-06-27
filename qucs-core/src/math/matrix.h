/*
 * matrix.h - matrix class definitions
 *
 * Copyright (C) 2003-2009 Stefan Jahn <stefan@lkcc.org>
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
 * the Free Software Foundation, Inc., 51 Franklin Street - Fifth Floor,
 * Boston, MA 02110-1301, USA.  
 *
 * $Id$
 *
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <functional>
#include <Eigen/Dense>
#include "complex.h"


#include "vector.h"

class matrix;

/*!\brief Dense complex matrix class */
class matrix
{
 public:
  /*!\brief Create an empty matrix

   Constructor creates an unnamed instance of the matrix class.
  */
  matrix () : m() {} ;
  /*!\brief Creates a square matrix

    Constructor creates an unnamed instance of the matrix class with a
    certain number of rows and columns.  Particularly creates a square matrix.  
    \param[in] s number of rows or colums of square matrix
  */
  matrix (const unsigned int s)  : m(s,s) {
    this->m.array().setZero();
  }
  
  /*! \brief Creates a matrix

   Constructor creates an unnamed instance of the matrix class with a
   certain number of rows and columns.  
   \param[in] r number of rows
   \param[in] c number of column
  */
  matrix (const unsigned int r, const unsigned int c)  : m(r,c) {
    this->m.array().setZero();
  }

  /*! \brief Create a matrix from an eigen object */
  matrix (const Eigen::Matrix<nr_complex_t,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> & rhs) : m(rhs) {}

  /*!\brief Assignment operator
    
    The assignment copy constructor creates a new instance based on the
    given matrix object. 
  
    \param[in] m object to copy
    \return assigned object
    \note m = m is safe
  */
  const matrix& operator = (const matrix & rhs) {
    if (&rhs != this)
      this->m = rhs.m;
    return *this;
  }

 
  /* copy constructor and destructor are now implicit thanks eigen */ 
  /* ~matrix (); */

  /*!\brief  Returns the matrix element at the given row and column.
   \param[in] r row number
   \param[in] c column number
   \todo delete and replace by ()
  */
  const nr_complex_t get (const unsigned int r, const unsigned int c) const {
    return this->m(r,c);
  }

  /*!\brief Sets the matrix element at the given row and column.
   \param[in] r row number
   \param[in] c column number
   \param[in] z complex number to assign
   \todo replace by ()
  */
  void set (const unsigned int r, const unsigned int c, const nr_complex_t z) {
    this->m(r,c) = z;
  }

  /*! \brief get number of columns
      \todo replace by cols() 
  */
  const unsigned int getCols (void) const { return this->m.cols(); }
  /*! \brief get number of rows
      \todo replace by rows() 
  */
  const unsigned int getRows (void) const { return this->m.rows(); }

  /*! get data aka linear array of coefficient */
  const nr_complex_t * getData (void) const { return this->m.data(); }

  void print (void) const;

  /*!\brief The function swaps the given rows with each other.
  \param[in] r1 source row
  \param[in] r2 destination row
  */
  void exchangeRows (const unsigned int r1, const unsigned int r2) {
    this->m.row(r1).swap(this->m.row(r2));
  }
  void exchangeCols (const unsigned int c1, const unsigned int c2) {
    this->m.col(c1).swap(this->m.col(c2));
  }

  // operator functions

  /*!\brief Matrix addition.
    \param[a] first matrix
    \param[b] second matrix
  */
  friend matrix operator + (const matrix & a, const matrix & b) {
    return matrix(a.m+b.m);
  }

  /*!\brief Complex scalar addition (complex+matrix)
   \param[in] b matrix 
   \param[in] z complex to add
  */
  friend matrix operator + (const nr_complex_t z, const matrix &b) {
    return matrix(z+b.m.array());
  }

  /*!\brief Complex scalar addition (matrix+complex)
   \param[in] b matrix 
   \param[in] z complex to add
  */
  friend matrix operator + (const matrix&b, const nr_complex_t z) {
    return z+b;
  }

  /*!\brief double scalar addition (double+matrix)
   \param[in] b matrix 
   \param[in] d double to add
  */
  friend matrix operator + (const nr_double_t d, const matrix &b) {
    return matrix(nr_complex_t(d)+b.m.array());
  }
 
  /*!\brief Complex scalar addition (matrix+double)
   \param[in] b matrix 
   \param[in] z complex to add
  */
  friend matrix operator + (const matrix&b, const nr_double_t d) {
    return d+b;
  }

  /*!\brief Matrix subtraction.
   \param[a] first matrix
   \param[b] second matrix
  */
  friend matrix operator - (const matrix &a, const matrix &b) {
    return matrix(a.m-b.m);
  }

   /*!\brief Complex scalar substraction (complex-matrix)
   \param[in] b matrix 
   \param[in] z complex to add
  */
  friend matrix operator - (const nr_complex_t z, const matrix &b) {
    return matrix(z-b.m.array());
  }

  /*!\brief Complex scalar addition (matrix-complex)
   \param[in] b matrix 
   \param[in] z complex to add
  */
  friend matrix operator - (const matrix&b, const nr_complex_t z) {
    return matrix(b.m.array()-z);
  }

  /*!\brief double scalar substraction (double-matrix)
   \param[in] b matrix 
   \param[in] d double to add
  */
  friend matrix operator - (const nr_double_t d, const matrix &b) {
    return matrix(nr_complex_t(d)-b.m.array());
  }
 
  /*!\brief Complex scalar addition (matrix-double)
   \param[in] b matrix 
   \param[in] z complex to add
  */
  friend matrix operator - (const matrix&b, const nr_double_t d) {
    return matrix(b.m.array()-nr_complex_t(d));
  }

  /*!\brief Matrix scaling division by complex version
   \param[in] a matrix to scale
   \param[in] d scaling real
   \return Scaled matrix
   \todo Ambiguous delete this operator
  */
  friend matrix operator / (const matrix &m, const nr_complex_t z) {
    return matrix((1.0/z)*m);
  }
   /*!\brief Matrix scaling division by real version
   \param[in] a matrix to scale
   \param[in] d scaling real
   \return Scaled matrix
   \todo Ambiguous delete this operator
  */
  friend matrix operator / (const matrix &m, const nr_double_t d) {
    return matrix((1.0/d)*m);
  }

  /*!\brief Matrix scaling complex version (complex*matrix)
   \param[in] a matrix to scale
   \param[in] z scaling complex
  */
  friend matrix operator * (const nr_complex_t z, const matrix &m) {
    return matrix(z*m);
  }

  /*!\brief Matrix scaling complex version (matrix*complex)
   \param[in] a matrix to scale
   \param[in] z scaling complex
  */
  friend matrix operator * (const matrix &m, const nr_complex_t z) {
    return z*m;
  }
  
  /*!\brief Matrix scaling double version (double*matrix)
   \param[in] a matrix to scale
   \param[in] d scaling real
   \return Scaled matrix
  */
  friend matrix operator * (const nr_double_t d, const matrix &m) {
    return matrix(d*m);
  }

  /*!\brief Matrix scaling double version (matrix*double)
   \param[in] a matrix to scale
   \param[in] d scaling real
   \return Scaled matrix
  */
  friend matrix operator * (const matrix &m, const nr_double_t d) {
    return d*m;
  }

  /*! Matrix multiplication.
    \param[a] first matrix
    \param[b] second matrix
    \note assert compatibility
  */
  friend matrix operator * (const matrix &a, const matrix &b) {
    return matrix(a.m*b.m);
  }

  // intrinsic operator functions

  /*!\brief Unary minus. */
  matrix operator  - () {
    return matrix(-this->m);
  }
 

  // other operations
  /*!\brief Matrix transposition
    \param[in] a Matrix to transpose
    \todo add transpose in place
  */
  friend matrix transpose (const matrix &a) {
    return matrix(a.m.transpose());
  }
  /*!\brief Conjugate complex matrix.
    \param[in] a Matrix to conjugate
  */
  friend matrix conj (const matrix &a) {
    return matrix(a.m.conjugate());
  }
  /*!\brief Computes magnitude of each matrix element.
   \param[in] a matrix
   \todo check for abs in place 
  */
  friend matrix abs (const matrix &a) {
    return matrix(a.m.array().abs().cast<nr_complex_t>());
  }

  /*!\brief Computes magnitude in dB of each matrix element.
   \param[in] a matrix
   \todo port to cwise operator
  */   
  friend matrix dB (const matrix &a) {
    return matrix(
	     a.m.unaryExpr(
	       std::pointer_to_unary_function<nr_complex_t,nr_double_t>(dB))
	       .cast<nr_complex_t>());
  }
  
  /*!\brief Computes the argument of each matrix element.
    \param[in] a matrix
   \todo port to cwise operator 
  */
  friend matrix arg (const matrix &a) {
    return matrix(
	     a.m.unaryExpr(
	       std::pointer_to_unary_function<const nr_complex_t&,nr_double_t>(std::arg))
	       .cast<nr_complex_t>());
  }

  /*!\brief adjoint matrix
   
   The function returns the adjoint complex matrix.  This is also
   called the adjugate or transpose conjugate. 
   \param[in] a Matrix to transpose
   \todo add adjoint in place
  */
  friend matrix adjoint (const matrix &a) {
     return matrix(a.m.adjoint());
  }

  /*!\brief Real part matrix.
     \param[in] a matrix
  */
  friend matrix real (const matrix &a) {
    return 
      matrix(
       (
	a.m.unaryExpr
	(
 	  std::ptr_fun(std::real<nr_complex_t>))
        ).cast<nr_complex_t>()
     );
  }

  /*!\brief Imaginary part matrix.
     \param[in] a matrix
  */
  friend matrix imag (const matrix &a) {
     return 
       matrix(
	(
         a.m.unaryExpr
         (
	  std::ptr_fun(std::imag<nr_complex_t>)
	 )
	).cast<nr_complex_t>()
      );
  }
  
  /*!\brief Create identity matrix with specified number of rows and columns.
   \param[in] rs row number
   \param[in] cs column number
   \todo rename to static version Identity()
  */
  
  friend matrix diagonal (class ::vector &);

  /*!\brief compute the power (integer) of a matrix
     \todo audit and port to eigen routine
  */
  friend matrix pow (const matrix &a, const int n);

  friend nr_complex_t cofactor (const matrix &a, const unsigned int, const unsigned int);
  friend nr_complex_t detLaplace (const matrix &a);
  friend nr_complex_t detGauss (const matrix &a);
  friend nr_complex_t det (matrix);
  friend matrix inverseLaplace (const matrix &a);
  friend matrix inverseGaussJordan (const matrix &a);
  friend matrix inverse (const matrix &a);
  friend matrix stos (matrix, nr_complex_t, nr_complex_t z0 = 50.0);
  friend matrix stos (matrix, nr_double_t, nr_double_t z0 = 50.0);
  friend matrix stos (matrix, ::vector, nr_complex_t z0 = 50.0);
  friend matrix stos (matrix, nr_complex_t, ::vector);
  friend matrix stos (matrix, ::vector, ::vector);
  friend matrix stoz (matrix, nr_complex_t z0 = 50.0);
  friend matrix stoz (matrix, ::vector);
  friend matrix ztos (matrix, nr_complex_t z0 = 50.0);
  friend matrix ztos (matrix, ::vector);
  friend matrix ztoy (matrix);
  friend matrix stoy (matrix, nr_complex_t z0 = 50.0);
  friend matrix stoy (matrix, ::vector);
  friend matrix ytos (matrix, nr_complex_t z0 = 50.0);
  friend matrix ytos (matrix, ::vector);
  friend matrix ytoz (matrix);
  friend matrix stoa (matrix, nr_complex_t z1 = 50.0, nr_complex_t z2 = 50.0);
  friend matrix atos (matrix, nr_complex_t z1 = 50.0, nr_complex_t z2 = 50.0);
  friend matrix stoh (matrix, nr_complex_t z1 = 50.0, nr_complex_t z2 = 50.0);
  friend matrix htos (matrix, nr_complex_t z1 = 50.0, nr_complex_t z2 = 50.0);
  friend matrix stog (matrix, nr_complex_t z1 = 50.0, nr_complex_t z2 = 50.0);
  friend matrix gtos (matrix, nr_complex_t z1 = 50.0, nr_complex_t z2 = 50.0);
  friend matrix cytocs (matrix, matrix);
  friend matrix cztocs (matrix, matrix);
  friend matrix cztocy (matrix, matrix);
  friend matrix cstocy (matrix, matrix);
  friend matrix cytocz (matrix, matrix);
  friend matrix cstocz (matrix, matrix);

  friend matrix twoport (matrix, char, char);
  friend nr_double_t rollet (matrix);
  friend nr_double_t b1 (matrix);

  /*! \brief Read access operator 
      \param[in] r: row number (from 0 like usually in C)
      \param[in] c: column number (from 0 like usually in C)
      \return Cell in the row r and column c
      \todo: Why not inline
      \todo: Why not r and c not const
      \todo: Create a debug version checking out of bound (using directly assert)
  */  
  nr_complex_t  operator () (const unsigned int r, const unsigned int c) const { 
    return this->m(r,c); 
  }
  /*! \brief Write access operator 
      \param[in] r: row number (from 0 like usually in C)
      \param[in] c: column number (from 0 like usually in C)
      \return Reference to cell in the row r and column c
      \todo: Why not inline
      \todo: Why r and c not const
      \todo: Create a debug version checking out of bound (using directly assert)
  */  
  nr_complex_t& operator () (const unsigned int r, const unsigned int c) { 
    return this->m(r,c); 
  }

 private:
  /*! Matrix data 
     \todo do not specify order and audit algorithm to be order agnostic (will allow to use lapack) 
  */
  Eigen::Matrix<nr_complex_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> m;
};

static inline matrix eye (const unsigned int rs, const unsigned int cs) {
    return matrix(Eigen::Matrix<nr_complex_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>::Identity(rs,cs));
}

  /*!\brief Create a square identity matrix
   \param[in] s row or column number of square matrix
   \todo rename to static version Identity()
  */
static inline matrix eye (const unsigned int s) {
    return matrix(Eigen::Matrix<nr_complex_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>::Identity(s,s));
  }


#endif /* __MATRIX_H__ */
