#! /bin/sh
#
# autogen.sh
#
# Run this script to re-generate all maintainer-generated files.
#
# Copyright (C) 2003 Stefan Jahn <stefan@lkcc.org>
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this package; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street - Fifth Floor,
# Boston, MA 02110-1301, USA.  
#

here=`pwd`
cd `dirname $0`

echo -n "Creating aclocal.m4... "
${ACLOCAL:-aclocal}
echo "done."
echo -n "Creating config.h.in... "
autoheader
echo "done."
echo -n "Libtoolizing... "
case `uname` in
  *Darwin*) LIBTOOLIZE=glibtoolize ;;
  *)        LIBTOOLIZE=libtoolize ;;
esac
$LIBTOOLIZE
echo "done."
echo -n "Creating Makefile.in(s)... "
${AUTOMAKE:-automake}
echo "done."
echo -n "Creating configure... "
autoconf
echo "done."

#
# run configure, maybe with parameters recorded in config.status
#
if [ -r config.status ]; then
  # Autoconf 2.13
  CMD=`awk '/^#.*\/?configure .*/ { $1 = ""; print; exit }' < config.status`
  if test "x$CMD" = "x" ; then
    # Autoconf 2.5x
    CMD=`grep "with options" < config.status | \
         sed 's/[^"]*["]\([^"]*\)["]/\1/' | sed 's/\\\//g' | sed "s/'//g"`
    CMD="./configure $CMD"
  fi
else
  CMD="./configure --enable-maintainer-mode"
fi
echo "Running $CMD $@ ..."
$CMD "$@"
