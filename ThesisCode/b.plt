#
# $Id: learningWithNoise.script,v 1.1 2002/11/12 01:57:16 simon Exp $
#
# Requires data files "[123].dat" from this directory,
# so change current working directory to this directory before running.
# gnuplot> set term <term-type>
# gnuplot> load 'simple.dem'
#
set output

type=2 ### 1 == pict,    0 == latex, 2 == win



if(type == 0) set term pslatex; set output 'learningWithNoise.tex'

##the next line is to see what is going on in a pict.  Also useful to save for viewing in PDF format.
if(type == 1) set term pict landscape color dashes "Monaco" 10 500 375
if(type == 2) set term win 

#no need to set range let the plot find it itself
#set yrange [0:0.5]
#set xrange [-0.5:49.5]

if(type == 0) set title '$\%$ recall vs position, with Hebbian learing in a 64 unit network with weight decay of $\%10$'
if((type == 1)||(type==2)) set title '% recall vs position with Hebbian learing in a 64 unit  network with weight decay of $\%10$'
set xlabel 'Pattern Position'
set ylabel '$\%$ Recall'


set tics out 
set grid ytics
set xtics 0,8,63
if(type == 0)set format x "$%g$";set format y "$%g$"
if((type == 1)||(type==2))set format x "%g";set format y "%g"

#this is how to set a key

set key top right spacing 0.8 width +2 box
set boxwidth 0.8

plot 'posRecallWdecay.txt' index 0:0:1 title "1" with lines,'posRecallWdecay.txt' index 3:3:1 title "4" with lines, 'posRecallWdecay.txt' index 7:7:1 title "8" with lines, 'posRecallWdecay.txt' index 11:11:1 title "12" with lines, 'posRecallWdecay.txt' index 15:15:1 title "16" with lines, 'posRecallWdecay.txt' index 19:19:1 title "20" with lines, 'posRecallWdecay.txt' index 23:23:1 title "24" with lines, 'posRecallWdecay.txt' index 27:27:1 title "28" with lines

# this is how to tilte things 
# plot [-10:+10] 'stabProfile.dat' title '$\mathcal{N}$' with line, '3.dat' title '$\mathcal{M}$odel' with line 


