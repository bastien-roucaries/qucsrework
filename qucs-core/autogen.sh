#! /bin/sh
#
# autogen.sh
#
# Run this script to re-generate all maintainer-generated files.
#
# Copyright (C) 2003, 2009 Stefan Jahn <stefan@lkcc.org>
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

if [ -d "./adms" ]; then
# if present, run autogen on the adms subproject
  if [ -e "./adms/autogen_lin.sh" ]; then
    ./adms/autogen_lin.sh "$@"
  elif [ -e "./adms/autogen.sh" ]; then
    ./adms/autogen.sh "$@";
  else
    echo "Could not locate adms autogen script in ./adms, you may use --disable-adms to use installed version"
    exit
  fi
else
  echo "No local adms source folder found (you may need to use the --diable-adms option to build with installed version)"
fi

autoreconf -f -i

#
# run configure, maybe with parameters recorded in config.status
#
if [ -r config.status ]; then
  # Autoconf 2.13
  CMD=`awk '/^#.*\/?configure .*/ { $1 = ""; print; exit }' < config.status`
  if test "x$CMD" = "x" ; then
    # Autoconf 2.5x
    eval set -- ./configure  `grep "with options" < config.status | \
         sed 's/^[^"]*["]\(.*\)\\\"$/\1/'` '"$@"'
  else
    set -- $CMD "$@"
  fi
else
  set -- ./configure --enable-maintainer-mode "$@"
fi
echo Running `for i; do echo "'$i'"$@; done` ...
"$@" --with-mkadms=internal
x