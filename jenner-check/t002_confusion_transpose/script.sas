/* Confusion-matrix pivot from utl-altair-slc-probabilistic-neural-networks-pnn-gaussain-kernel-using-r.sas */
/* In the original, WORKX.CONFUSION_DF is produced by the R (PNN) step. Here it is rebuilt as a WORK data  */
/* step from the long-form confusion table the program documents in its own output, so the identical        */
/* PROC TRANSPOSE (by pred_species / id true_species / var count) runs standalone and reproduces            */
/* WORKX.CONFUSION_XPO.                                                                                      */

data work.confusion_df;
 length True_Species Pred_Species $10;
 input True_Species $ Pred_Species $ Count;
cards;
setosa      setosa     13
versicolor  setosa      0
virginica   setosa      0
setosa      versicolor  0
versicolor  versicolor 12
virginica   versicolor  0
setosa      virginica   0
versicolor  virginica   1
virginica   virginica  13
;
run;

proc transpose data=work.confusion_df out=work.confusion_xpo;
 by pred_species;
 id true_species;
 var count;
run;quit;

proc print data=work.confusion_xpo noobs;
run;
