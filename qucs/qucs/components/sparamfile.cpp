/***************************************************************************
                          sparamfile.cpp  -  description
                             -------------------
    begin                : Sat Aug 23 2003
    copyright            : (C) 2003 by Michael Margraf
    email                : margraf@mwt.ee.tu-berlin.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "sparamfile.h"


SParamFile::SParamFile(int No)
{
  Description = "S parameter file";

  int h = 30*((No-1)/2) + 15;
  Lines.append(new Line(-15, -h, 15, -h,QPen(QPen::darkBlue,2)));
  Lines.append(new Line( 15, -h, 15,  h,QPen(QPen::darkBlue,2)));
  Lines.append(new Line(-15,  h, 15,  h,QPen(QPen::darkBlue,2)));
  Lines.append(new Line(-15, -h,-15,  h,QPen(QPen::darkBlue,2)));
  Texts.append(new Text( -6,  0,"file"));


  int i=0, y = 15-h;
  while(i<No) {
    i++;
    Lines.append(new Line(-30,  y,-15,  y,QPen(QPen::darkBlue,2)));
    Ports.append(new Port(-30,  y));
    Texts.append(new Text(-25,y-3,QString::number(i)));

    if(i == No) break;
    i++;
    Lines.append(new Line( 15,  y, 30,  y,QPen(QPen::darkBlue,2)));
    Ports.append(new Port( 30,  y));
    Texts.append(new Text( 20,y-3,QString::number(i)));
    y += 60;
  }

  Lines.append(new Line( 0, h, 0,h+15,QPen(QPen::darkBlue,2)));
  Texts.append(new Text( 4,h+10,"Ref"));
  Ports.append(new Port( 0,h+15));    // 'Ref' port

  x1 = -30; y1 = -h-2;
  x2 =  30; y2 =  h+15;

  tx = x1+4;
  ty = y2+4;
  Sign  = QString("SPfile")+QString::number(No);
  Model = QString("SPfile")+QString::number(No);
  Name  = "X";

  Props.append(new Property("File", "test.s2p", true, "name of the s parameter file"));
}

SParamFile::~SParamFile()
{
}

SParamFile* SParamFile::newOne()
{
  int z = Sign.mid(6).toInt();
  return new SParamFile(z);
}