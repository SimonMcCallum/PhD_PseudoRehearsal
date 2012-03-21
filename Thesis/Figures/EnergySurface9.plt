#
# $Id: signGraph.script,v 1.1 2002/11/13 03:17:50 simon Exp $
#
# Requires data files "[123].dat" from this directory,
# so change current working directory to this directory before running.
# gnuplot> set term <term-type>
# gnuplot> load 'simple.dem'
#
set output

type=0 ### 1 == pict,    0 == latex


if(type == 0) set term pslatex size 5, 3; set output 'EnergySurface16_0_1.tex' 
##the next line is to see what is going on in a pict.  Also useful to save for viewing in PDF format.
if(type == 1) set term png; set output 'EnergySurface16_0_1.png'
if(type == 2) set term x11 

#no need to set range let the plot find it itself

set border
#set clip
#set nopolar
set yrange [0:21]
set xrange [0:21]
set xtics border nomirror 1
set ytics border nomirror offset 1 1
set ztics border nomirror 1

if(type == 0) set title 'Energy Surface.'
if(type == 1) set title 'Energy Surface.'
set xlabel ''
set ylabel ''

set format z "%3.1f"
if(type == 0)set format x "$%g$";set format y "$%g$"
if(type == 1)set format x "%g";set format y "%g"

#this is how to set a key
set key 0,0,-1 spacing 1.6

set surface
set hidden3d

splot 'EnergySurface9noise_1_0.dat' using 4:3:5 every 1 with lines 

set term win
replot

pause -1

splot 'EnergySurface9_1_0.dat' using 4:3:5 every 1 with lines 


reset

set surface
set hidden3d

#sgn(x),
