/***************************************************************************
                                 spicefile.h
                                -------------
    begin                : Tue Dec 28 2004
    copyright            : (C) 2004 by Michael Margraf
    email                : michael.margraf@alumni.tu-berlin.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef SPICEFILE_H
#define SPICEFILE_H
#include <QtGui>
#include "component.h"

#include <qobject.h>
#include <qdatetime.h>
//Added by qt3to4:
#include <Q3TextStream>

class Q3Process;
class Q3TextStream;
class QString;

class SpiceFile : public QObject, public MultiViewComponent  {
   Q_OBJECT
public:
  SpiceFile();
 ~SpiceFile() {};
  Component* newOne();
  static Element* info(QString&, char* &, bool getNewOne=false);

  bool withSim;
  bool createSubNetlist(QTextStream *);
  QString getErrorText() { return ErrText; }
  QString getSubcircuitFile();

private:
  bool makeSubcircuit;
  bool insertSim;
  bool changed;
  Q3Process *QucsConv, *SpicePrep;
  QString NetText, ErrText, NetLine, SimText;
  QTextStream *outstream, *filstream, *prestream;
  QDateTime lastLoaded;
  bool recreateSubNetlist(QString *, QString *);

protected:
  QString netlist();
  void createSymbol();

private slots:
  void slotGetNetlist();
  void slotGetError();
  void slotExited();
  void slotSkipOut();
  void slotSkipErr();
  void slotGetPrepOut();
  void slotGetPrepErr();
};

#endif
