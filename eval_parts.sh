for r in .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0 ;
do
    b eval_part.slurm 0 $r $1
done