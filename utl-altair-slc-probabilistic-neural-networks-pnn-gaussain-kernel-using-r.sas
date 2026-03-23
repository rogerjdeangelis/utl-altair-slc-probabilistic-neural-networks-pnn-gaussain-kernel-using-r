%let pgm=utl-altair-slc-probabilistic-neural-networks-pnn-gaussain-kernel-using-r;

Altair slc probabilistic neural networks pnn gaussain kernel using r

Too long to post on a list, see github
https://github.com/rogerjdeangelis/utl-altair-slc-probabilistic-neural-networks-pnn-gaussain-kernel-using-r

community  post
https://community.altair.com/discussion/58644/pnn-implementation?tab=accepted

Confirms PNN result
Multidimensional Scaling to reveal underlying structures in the data.
https://github.com/rogerjdeangelis/altair-slc-r-simple-random-forest-classification-example-using-iris-dataset

                        Dimension 1
        -5.0     -2.5      0.0      2.5      5.0
      2 +-+--------+--------+--------+--------+--+ 2
        |                                        |
        |          +    + Setosa        X        |
   D    |         +     . Versicolor     X       |   D
   I  1 +          +    X Virginica              + 1 I
   M    |         +++                  X         |   M
   E    |          +             .   X XX        |   E
   N    |         +++          ..  XXXXX X       |   N
   S    |       + ++           ....XXX    X      |   S
   I  0 +         +++        .... XXXX           + 0 I
   O    |         ++       . ....XXXX            |   O
   N    |        ++        ..... XX              |   N
        |       + +        .....XXX              |
   2    |                .  ..   X               |   2
     -1 +         +      .                       + 1
        |                 .   X                  |
        |                                        |
        +-+--------+--------+--------+--------+--+
        -5.0     -2.5      0.0      2.5      5.0
                       Dimension 1

WHAT WE WANT
PROBLEM CLASSIFICATION BASED ON A NEURAL NET (ALMOST PERFECT)

+---------------------------------------------------------------+
|  TESTING (HOLD OUT) SAMPLE                                    |
|  WORKX.CONFUSION_XPO total obs=3 ALMOST PERFECT               |
|                                                               |
|  PRED_                                                        |
|  SPECIES       COUNT     setosa    versicolor    virginica    |
|                                                               |
|  setosa        COUNT       13           0             0       |
|  versicolor    COUNT        0          12             0       |
|  virginica     COUNT        0           1            13       |
+---------------------------------------------------------------+

/*                   _
(_)_ __  _ __  _   _| |_
| | `_ \| `_ \| | | | __|
| | | | | |_) | |_| | |_
|_|_| |_| .__/ \__,_|\__|
        |_|
*/

options validvarname=v7;
data workx.iris;
 retain Sepal_Length Sepal_Width Petal_Length Petal_Width Species ;
 informat Species $11.;
 input
 Sepal_Length Sepal_Width Petal_Length Petal_Width Species @@;
cards4;
5.1 3.5 1.4 0.2 setosa 7.0 3.2 4.7 1.4 versicolor 6.3 3.3 6.0 2.5 virginica
4.9 3.0 1.4 0.2 setosa 6.4 3.2 4.5 1.5 versicolor 5.8 2.7 5.1 1.9 virginica
4.7 3.2 1.3 0.2 setosa 6.9 3.1 4.9 1.5 versicolor 7.1 3.0 5.9 2.1 virginica
4.6 3.1 1.5 0.2 setosa 5.5 2.3 4.0 1.3 versicolor 6.3 2.9 5.6 1.8 virginica
5.0 3.6 1.4 0.2 setosa 6.5 2.8 4.6 1.5 versicolor 6.5 3.0 5.8 2.2 virginica
5.4 3.9 1.7 0.4 setosa 5.7 2.8 4.5 1.3 versicolor 7.6 3.0 6.6 2.1 virginica
4.6 3.4 1.4 0.3 setosa 6.3 3.3 4.7 1.6 versicolor 4.9 2.5 4.5 1.7 virginica
5.0 3.4 1.5 0.2 setosa 4.9 2.4 3.3 1.0 versicolor 7.3 2.9 6.3 1.8 virginica
4.4 2.9 1.4 0.2 setosa 6.6 2.9 4.6 1.3 versicolor 6.7 2.5 5.8 1.8 virginica
4.9 3.1 1.5 0.1 setosa 5.2 2.7 3.9 1.4 versicolor 7.2 3.6 6.1 2.5 virginica
5.4 3.7 1.5 0.2 setosa 5.0 2.0 3.5 1.0 versicolor 6.5 3.2 5.1 2.0 virginica
4.8 3.4 1.6 0.2 setosa 5.9 3.0 4.2 1.5 versicolor 6.4 2.7 5.3 1.9 virginica
4.8 3.0 1.4 0.1 setosa 6.0 2.2 4.0 1.0 versicolor 6.8 3.0 5.5 2.1 virginica
4.3 3.0 1.1 0.1 setosa 6.1 2.9 4.7 1.4 versicolor 5.7 2.5 5.0 2.0 virginica
5.8 4.0 1.2 0.2 setosa 5.6 2.9 3.6 1.3 versicolor 5.8 2.8 5.1 2.4 virginica
5.7 4.4 1.5 0.4 setosa 6.7 3.1 4.4 1.4 versicolor 6.4 3.2 5.3 2.3 virginica
5.4 3.9 1.3 0.4 setosa 5.6 3.0 4.5 1.5 versicolor 6.5 3.0 5.5 1.8 virginica
5.1 3.5 1.4 0.3 setosa 5.8 2.7 4.1 1.0 versicolor 7.7 3.8 6.7 2.2 virginica
5.7 3.8 1.7 0.3 setosa 6.2 2.2 4.5 1.5 versicolor 7.7 2.6 6.9 2.3 virginica
5.1 3.8 1.5 0.3 setosa 5.6 2.5 3.9 1.1 versicolor 6.0 2.2 5.0 1.5 virginica
5.4 3.4 1.7 0.2 setosa 5.9 3.2 4.8 1.8 versicolor 6.9 3.2 5.7 2.3 virginica
5.1 3.7 1.5 0.4 setosa 6.1 2.8 4.0 1.3 versicolor 5.6 2.8 4.9 2.0 virginica
4.6 3.6 1.0 0.2 setosa 6.3 2.5 4.9 1.5 versicolor 7.7 2.8 6.7 2.0 virginica
5.1 3.3 1.7 0.5 setosa 6.1 2.8 4.7 1.2 versicolor 6.3 2.7 4.9 1.8 virginica
4.8 3.4 1.9 0.2 setosa 6.4 2.9 4.3 1.3 versicolor 6.7 3.3 5.7 2.1 virginica
5.0 3.0 1.6 0.2 setosa 6.6 3.0 4.4 1.4 versicolor 7.2 3.2 6.0 1.8 virginica
5.0 3.4 1.6 0.4 setosa 6.8 2.8 4.8 1.4 versicolor 6.2 2.8 4.8 1.8 virginica
5.2 3.5 1.5 0.2 setosa 6.7 3.0 5.0 1.7 versicolor 6.1 3.0 4.9 1.8 virginica
5.2 3.4 1.4 0.2 setosa 6.0 2.9 4.5 1.5 versicolor 6.4 2.8 5.6 2.1 virginica
4.7 3.2 1.6 0.2 setosa 5.7 2.6 3.5 1.0 versicolor 7.2 3.0 5.8 1.6 virginica
4.8 3.1 1.6 0.2 setosa 5.5 2.4 3.8 1.1 versicolor 7.4 2.8 6.1 1.9 virginica
5.4 3.4 1.5 0.4 setosa 5.5 2.4 3.7 1.0 versicolor 7.9 3.8 6.4 2.0 virginica
5.2 4.1 1.5 0.1 setosa 5.8 2.7 3.9 1.2 versicolor 6.4 2.8 5.6 2.2 virginica
5.5 4.2 1.4 0.2 setosa 6.0 2.7 5.1 1.6 versicolor 6.3 2.8 5.1 1.5 virginica
4.9 3.1 1.5 0.2 setosa 5.4 3.0 4.5 1.5 versicolor 6.1 2.6 5.6 1.4 virginica
5.0 3.2 1.2 0.2 setosa 6.0 3.4 4.5 1.6 versicolor 7.7 3.0 6.1 2.3 virginica
5.5 3.5 1.3 0.2 setosa 6.7 3.1 4.7 1.5 versicolor 6.3 3.4 5.6 2.4 virginica
4.9 3.6 1.4 0.1 setosa 6.3 2.3 4.4 1.3 versicolor 6.4 3.1 5.5 1.8 virginica
4.4 3.0 1.3 0.2 setosa 5.6 3.0 4.1 1.3 versicolor 6.0 3.0 4.8 1.8 virginica
5.1 3.4 1.5 0.2 setosa 5.5 2.5 4.0 1.3 versicolor 6.9 3.1 5.4 2.1 virginica
5.0 3.5 1.3 0.3 setosa 5.5 2.6 4.4 1.2 versicolor 6.7 3.1 5.6 2.4 virginica
4.5 2.3 1.3 0.3 setosa 6.1 3.0 4.6 1.4 versicolor 6.9 3.1 5.1 2.3 virginica
4.4 3.2 1.3 0.2 setosa 5.8 2.6 4.0 1.2 versicolor 5.8 2.7 5.1 1.9 virginica
5.0 3.5 1.6 0.6 setosa 5.0 2.3 3.3 1.0 versicolor 6.8 3.2 5.9 2.3 virginica
5.1 3.8 1.9 0.4 setosa 5.6 2.7 4.2 1.3 versicolor 6.7 3.3 5.7 2.5 virginica
4.8 3.0 1.4 0.3 setosa 5.7 3.0 4.2 1.2 versicolor 6.7 3.0 5.2 2.3 virginica
5.1 3.8 1.6 0.2 setosa 5.7 2.9 4.2 1.3 versicolor 6.3 2.5 5.0 1.9 virginica
4.6 3.2 1.4 0.2 setosa 6.2 2.9 4.3 1.3 versicolor 6.5 3.0 5.2 2.0 virginica
5.3 3.7 1.5 0.2 setosa 5.1 2.5 3.0 1.1 versicolor 6.2 3.4 5.4 2.3 virginica
5.0 3.3 1.4 0.2 setosa 5.7 2.8 4.1 1.3 versicolor 5.9 3.0 5.1 1.8 virginica
;;;;
run;

proc sort data=workx.iris;
 by species;
run;

/**************************************************************************************************************************/
/*  WORKX.IRIS total obs=150                                                                                              */
/*        species_    sepal_    sepal_    petal_    petal_                                                                */
/* Obs      name      length     width    length     width    species                                                     */
/*                                                                                                                        */
/*   1     setosa       5.1       3.5       1.4       0.2        0                                                        */
/*   2     setosa       4.9       3.0       1.4       0.2        0                                                        */
/*   3     setosa       4.7       3.2       1.3       0.2        0                                                        */
/*   4     setosa       4.6       3.1       1.5       0.2        0                                                        */
/*   5     setosa       5.0       3.6       1.4       0.2        0                                                        */
/* ...                                                                                                                    */
/* 146     virginica    6.7       3.0       5.2       2.3        2                                                        */
/* 147     virginica    6.3       2.5       5.0       1.9        2                                                        */
/* 148     virginica    6.5       3.0       5.2       2.0        2                                                        */
/* 149     virginica    6.2       3.4       5.4       2.3        2                                                        */
/* 150 ;    virginica    5.9       3.0       5.1       1.8        2                                                        */
/**************************************************************************************************************************/

/*                   _     _
(_)_ __  _ __  _   _| |_  | | ___   __ _
| | `_ \| `_ \| | | | __| | |/ _ \ / _` |
| | | | | |_) | |_| | |_  | | (_) | (_| |
|_|_| |_| .__/ \__,_|\__| |_|\___/ \__, |
        |_|                        |___/
*/

1                                          Altair SLC          13:33 Monday, March 23, 2026

NOTE: Copyright 2002-2025 World Programming, an Altair Company
NOTE: Altair SLC 2026 (05.26.01.00.000758)
      Licensed to Roger DeAngelis
NOTE: This session is executing on the X64_WIN11PRO platform and is running in 64 bit mode

NOTE: AUTOEXEC processing beginning; file is C:\wpsoto\autoexec.sas
NOTE: AUTOEXEC source line
1       +  ï»¿ods _all_ close;
           ^

NOTE: AUTOEXEC processing completed

1         options set=RHOME "C:\Progra~1\R\R-4.5.2\bin\r";
2
3         proc r;
NOTE: Using R version 4.5.2 (2025-10-31 ucrt) from C:\Program Files\R\R-4.5.2
4         export data=workx.iris r=iris;
NOTE: Creating R data frame 'iris' from data set 'WORKX.iris'

5         submit;
6
7         # Make Species a factor FIRST (critical fix)
8
9         iris$Species <- factor(iris$Species, levels=c("setosa","versicolor","virginica"))
10
11        # STRATIFIED SPLIT - BALANCED CLASSES
12        split_by_class <- function(df, train_prop=0.75) {
13          train_idx <- c()
14          for(sp in levels(df$Species)) {
15            idx <- which(df$Species == sp)
16            n_sp <- length(idx)
17            n_train_sp <- floor(train_prop * n_sp)
18            train_idx <- c(train_idx, sample(idx, n_train_sp))
19          }
20          return(train_idx)
21        }
22
23        train_idx <- split_by_class(iris, 0.75)
24        train <- iris[train_idx,]
25        test <- iris[-train_idx,]  # Now works!
26
27        cat("Train class balance:\n")
28        print(table(train$Species))
29        cat("Test class balance:\n")
30        print(table(test$Species))
31
32        # Rest of your PNN code runs perfectly now...
33        X_train <- as.matrix(train[,1:4])
34        y_train <- as.numeric(train$Species) - 1
35        X_test <- as.matrix(test[,1:4])
36        y_test <- as.numeric(test$Species) - 1
37        K <- 3
38        species_names <- c("setosa","versicolor","virginica")
39
40        # Your PNN function (unchanged - perfect)
41        pnn_gaussian <- function(X_train, y_train, X_test, sigma=0.3) {
42          n_test <- nrow(X_test)
43          probs <- matrix(0.001, n_test, K)
44          for(i in 1:n_test) {
45            x <- X_test[i,]
46            for(k in 1:K) {
47              class_idx <- which(y_train == (k-1))
48              if(length(class_idx) == 0) next
49              X_k <- X_train[class_idx,]
50              dists_sq <- rowSums((X_k - x)^2)
51              kernel <- exp(-dists_sq / (2 * sigma^2))
52              probs[i,k] <- mean(kernel)
53            }
54            probs[i,] <- probs[i,] / sum(probs[i,])
55          }
56          pred_classes <- (apply(probs, 1, which.max) - 1)
57          return(list(pred=pred_classes, probs=probs))
58        }
59
60        sigma <- 0.25
61        result <- pnn_gaussian(X_train, y_train, X_test, sigma)
62        pred_classes <- result$pred
63
64        accuracy <- mean(pred_classes == y_test)
65        cat(sprintf("\n FIXED PNN Accuracy: %.1f%% (%d/%d)\n",
66              accuracy*100, sum(pred_classes==y_test), length(y_test)))
67
68        # CONFUSION MATRIX
69        cat("\n CONFUSION MATRIX:\n")
70        true_f <- factor(y_test, levels=0:2, labels=species_names)
71        pred_f <- factor(pred_classes, levels=0:2, labels=species_names)
72        print(table(True=true_f, Predicted=pred_f))
73
74        # Base confusion matrix table
75        cm_table <- table(True=true_f, Predicted=pred_f)
76
77        # Convert to proper dataframe with row/column names
78        confusion_df <- as.data.frame(as.table(cm_table))
79        colnames(confusion_df) <- c("True_Species", "Pred_Species", "Count")
80
81        confusion_df
82
83        # TOP 5 PREDICTIONS
84        cat("\n TOP 5:\n")
85        for(i in 1:5) {
86          true_sp <- species_names[y_test[i]+1]
87          pred_sp <- species_names[pred_classes[i]+1]
88          maxp <- round(max(result$probs[i,])*100, 1)
89          cat(sprintf(" %d: %s > %s (%.1f%%)\n", i, true_sp, pred_sp, maxp))
90        }
91
92        # NEW FLOWER
93        new_flower <- matrix(c(6.5,3.0,5.5,2.0), nrow=1)
94        new_res <- pnn_gaussian(X_train, y_train, new_flower, sigma)
95        cat(sprintf("\n NEW %s (%.1f%%)\n",
96              species_names[new_res$pred[1]+1], max(new_res$probs[1,])*100))
97
98        endsubmit;

NOTE: Submitting statements to R:

>
> # Make Species a factor FIRST (critical fix)
>
> iris$Species <- factor(iris$Species, levels=c("setosa","versicolor","virginica"))
>
> # STRATIFIED SPLIT - BALANCED CLASSES
> split_by_class <- function(df, train_prop=0.75) {
+   train_idx <- c()
+   for(sp in levels(df$Species)) {
+     idx <- which(df$Species == sp)
+     n_sp <- length(idx)
+     n_train_sp <- floor(train_prop * n_sp)
+     train_idx <- c(train_idx, sample(idx, n_train_sp))
+   }
+   return(train_idx)
+ }
>
> train_idx <- split_by_class(iris, 0.75)
> train <- iris[train_idx,]
> test <- iris[-train_idx,]  # Now works!
>
> cat("Train class balance:\n")
> print(table(train$Species))
> cat("Test class balance:\n")
> print(table(test$Species))
>
> # Rest of your PNN code runs perfectly now...
> X_train <- as.matrix(train[,1:4])
> y_train <- as.numeric(train$Species) - 1
> X_test <- as.matrix(test[,1:4])
> y_test <- as.numeric(test$Species) - 1
> K <- 3
> species_names <- c("setosa","versicolor","virginica")
>
> # Your PNN function (unchanged - perfect)
> pnn_gaussian <- function(X_train, y_train, X_test, sigma=0.3) {
+   n_test <- nrow(X_test)
+   probs <- matrix(0.001, n_test, K)
+   for(i in 1:n_test) {
+     x <- X_test[i,]
+     for(k in 1:K) {
+       class_idx <- which(y_train == (k-1))
+       if(length(class_idx) == 0) next
+       X_k <- X_train[class_idx,]
+       dists_sq <- rowSums((X_k - x)^2)
+       kernel <- exp(-dists_sq / (2 * sigma^2))
+       probs[i,k] <- mean(kernel)
+     }
+     probs[i,] <- probs[i,] / sum(probs[i,])
+   }
+   pred_classes <- (apply(probs, 1, which.max) - 1)
+   return(list(pred=pred_classes, probs=probs))
+ }
>
> sigma <- 0.25
> result <- pnn_gaussian(X_train, y_train, X_test, sigma)
> pred_classes <- result$pred
>
> accuracy <- mean(pred_classes == y_test)
> cat(sprintf("\n FIXED PNN Accuracy: %.1f%% (%d/%d)\n",
+       accuracy*100, sum(pred_classes==y_test), length(y_test)))
>
> # CONFUSION MATRIX
> cat("\n CONFUSION MATRIX:\n")
> true_f <- factor(y_test, levels=0:2, labels=species_names)
> pred_f <- factor(pred_classes, levels=0:2, labels=species_names)
> print(table(True=true_f, Predicted=pred_f))
>
> # Base confusion matrix table
> cm_table <- table(True=true_f, Predicted=pred_f)
>
> # Convert to proper dataframe with row/column names
> confusion_df <- as.data.frame(as.table(cm_table))
> colnames(confusion_df) <- c("True_Species", "Pred_Species", "Count")
>
> confusion_df
>
> # TOP 5 PREDICTIONS
> cat("\n TOP 5:\n")
> for(i in 1:5) {
+   true_sp <- species_names[y_test[i]+1]
+   pred_sp <- species_names[pred_classes[i]+1]
+   maxp <- round(max(result$probs[i,])*100, 1)
+   cat(sprintf(" %d: %s > %s (%.1f%%)\n", i, true_sp, pred_sp, maxp))
+ }
>
> # NEW FLOWER
> new_flower <- matrix(c(6.5,3.0,5.5,2.0), nrow=1)
> new_res <- pnn_gaussian(X_train, y_train, new_flower, sigma)
> cat(sprintf("\n NEW %s (%.1f%%)\n",
+       species_names[new_res$pred[1]+1], max(new_res$probs[1,])*100))
>

NOTE: Processing of R statements complete

99        import r=confusion_df data=workx.confusion_df;
NOTE: Creating data set 'WORKX.confusion_df' from R data frame 'confusion_df'
NOTE: Column names modified during import of 'confusion_df'
NOTE: Data set "WORKX.confusion_df" has 9 observation(s) and 3 variable(s)

100       import r=test data=workx.test;
NOTE: Creating data set 'WORKX.test' from R data frame 'test'
NOTE: Column names modified during import of 'test'
NOTE: Data set "WORKX.test" has 39 observation(s) and 5 variable(s)

101       import r=train data=workx.train;
NOTE: Creating data set 'WORKX.train' from R data frame 'train'
NOTE: Column names modified during import of 'train'
NOTE: Data set "WORKX.train" has 111 observation(s) and 5 variable(s)

102       run;
NOTE: Procedure r step took :
      real time : 0.475
      cpu time  : 0.078


103
104
ERROR: Error printed on page 1

NOTE: Submitted statements took :
      real time : 0.554
      cpu time  : 0.156


/*                                                                                      _
 _ __  _ __ ___   ___ ___  ___ ___   _ __  _ __  _ __    __ _  __ _ _   _ ___ ___  __ _(_)_ __
| `_ \| `__/ _ \ / __/ _ \/ __/ __| | `_ \| `_ \| `_ \  / _` |/ _` | | | / __/ __|/ _` | | `_ \
| |_) | | | (_) | (_|  __/\__ \__ \ | |_) | | | | | | || (_| | (_| | |_| \__ \__ \ (_| | | | | |
| .__/|_|  \___/ \___\___||___/___/ | .__/|_| |_|_| |_| \__, |\__,_|\__,_|___/___/\__,_|_|_| |_|
|_|                                 |_|                 |___/

*/

options set=RHOME "C:\Progra~1\R\R-4.5.2\bin\r";

proc r;
export data=workx.iris r=iris;
submit;

# Make Species a factor FIRST

iris$Species <- factor(iris$Species, levels=c("setosa","versicolor","virginica"))

# STRATIFIED SPLIT - BALANCED CLASSES
split_by_class <- function(df, train_prop=0.75) {
  train_idx <- c()
  for(sp in levels(df$Species)) {
    idx <- which(df$Species == sp)
    n_sp <- length(idx)
    n_train_sp <- floor(train_prop * n_sp)
    train_idx <- c(train_idx, sample(idx, n_train_sp))
  }
  return(train_idx)
}

train_idx <- split_by_class(iris, 0.75)
train <- iris[train_idx,]
test <- iris[-train_idx,]  # Now works!

cat("Train class balance:\n")
print(table(train$Species))
cat("Test class balance:\n")
print(table(test$Species))

# Rest of your PNN code runs perfectly now...
X_train <- as.matrix(train[,1:4])
y_train <- as.numeric(train$Species) - 1
X_test <- as.matrix(test[,1:4])
y_test <- as.numeric(test$Species) - 1
K <- 3
species_names <- c("setosa","versicolor","virginica")

# Your PNN function (unchanged - perfect)
pnn_gaussian <- function(X_train, y_train, X_test, sigma=0.3) {
  n_test <- nrow(X_test)
  probs <- matrix(0.001, n_test, K)
  for(i in 1:n_test) {
    x <- X_test[i,]
    for(k in 1:K) {
      class_idx <- which(y_train == (k-1))
      if(length(class_idx) == 0) next
      X_k <- X_train[class_idx,]
      dists_sq <- rowSums((X_k - x)^2)
      kernel <- exp(-dists_sq / (2 * sigma^2))
      probs[i,k] <- mean(kernel)
    }
    probs[i,] <- probs[i,] / sum(probs[i,])
  }
  pred_classes <- (apply(probs, 1, which.max) - 1)
  return(list(pred=pred_classes, probs=probs))
}

sigma <- 0.25
result <- pnn_gaussian(X_train, y_train, X_test, sigma)
pred_classes <- result$pred

accuracy <- mean(pred_classes == y_test)
cat(sprintf("\n FIXED PNN Accuracy: %.1f%% (%d/%d)\n",
      accuracy*100, sum(pred_classes==y_test), length(y_test)))

# CONFUSION MATRIX
cat("\n CONFUSION MATRIX:\n")
true_f <- factor(y_test, levels=0:2, labels=species_names)
pred_f <- factor(pred_classes, levels=0:2, labels=species_names)
print(table(True=true_f, Predicted=pred_f))

# Base confusion matrix table
cm_table <- table(True=true_f, Predicted=pred_f)

# Convert to proper dataframe with row/column names
confusion_df <- as.data.frame(as.table(cm_table))
colnames(confusion_df) <- c("True_Species", "Pred_Species", "Count")

confusion_df

# TOP 5 PREDICTIONS
cat("\n TOP 5:\n")
for(i in 1:5) {
  true_sp <- species_names[y_test[i]+1]
  pred_sp <- species_names[pred_classes[i]+1]
  maxp <- round(max(result$probs[i,])*100, 1)
  cat(sprintf(" %d: %s > %s (%.1f%%)\n", i, true_sp, pred_sp, maxp))
}

# NEW FLOWER
new_flower <- matrix(c(6.5,3.0,5.5,2.0), nrow=1)
new_res <- pnn_gaussian(X_train, y_train, new_flower, sigma)
cat(sprintf("\n NEW %s (%.1f%%)\n",
      species_names[new_res$pred[1]+1], max(new_res$probs[1,])*100))

endsubmit;
import r=confusion_df data=workx.confusion_df;
import r=test data=workx.test;
import r=train data=workx.train;
run;

proc transpose data=workx.confusion_df out=workx.confusion_xpo;
 by pred_species;
 id true_species;
 var count;
run;quit;

/*           _               _
  ___  _   _| |_ _ __  _   _| |_
 / _ \| | | | __| `_ \| | | | __|
| (_) | |_| | |_| |_) | |_| | |_
 \___/ \__,_|\__| .__/ \__,_|\__|
                |_|
*/

/**************************************************************************************************************************/
/*                                                                                                                        */
/* WORKX.CONFUSION_XPO total obs=3 23MAR2026:13:42:10                                                                     */
/*                                                                                                                        */
/* PRED_                                                                                                                  */
/* SPECIES       _NAME_    setosa    versicolor    virginica                                                              */
/*                                                                                                                        */
/* setosa        COUNT       13           0             0                                                                 */
/* versicolor    COUNT        0          12             0                                                                 */
/* virginica     COUNT        0           1            13                                                                 */
/*                                                                                                                        */
/* BALANCED                                                                                                               */
/*                                                                                                                        */
/* Train class balance:                                                                                                   */
/*     setosa versicolor  virginica                                                                                       */
/*         37         37         37                                                                                       */
/*                                                                                                                        */
/* Test class balance:                                                                                                    */
/*     setosa versicolor  virginica                                                                                       */
/*         13         13         13                                                                                       */
/*                                                                                                                        */
/*                                                                                                                        */
/*  FIXED PNN Accuracy: 97.4% (38/39)                                                                                     */
/*  CONFUSION MATRIX:                                                                                                     */
/*             Predicted                                                                                                  */
/* True         setosa versicolor virginica                                                                               */
/*   setosa         13          0         0                                                                               */
/*   versicolor      0         12         1                                                                               */
/*   virginica       0          0        13                                                                               */
/*                                                                                                                        */
/*   True_Species Pred_Species Count                                                                                      */
/* 1       setosa       setosa    13                                                                                      */
/* 2   versicolor       setosa     0                                                                                      */
/* 3    virginica       setosa     0                                                                                      */
/* 4       setosa   versicolor     0                                                                                      */
/* 5   versicolor   versicolor    12                                                                                      */
/* 6    virginica   versicolor     0                                                                                      */
/* 7       setosa    virginica     0                                                                                      */
/* 8   versicolor    virginica     1                                                                                      */
/* 9    virginica    virginica    13                                                                                      */
/*                                                                                                                        */
/*  TOP 5:                                                                                                                */
/*  1: setosa > setosa (100.0%)                                                                                           */
/*  2: setosa > setosa (100.0%)                                                                                           */
/*  3: setosa > setosa (100.0%)                                                                                           */
/*  4: setosa > setosa (100.0%)                                                                                           */
/*  5: setosa > setosa (100.0%)                                                                                           */
/**************************************************************************************************************************/
/*                                   _
 _ __  _ __ ___   ___ ___  ___ ___  | | ___   __ _
| `_ \| `__/ _ \ / __/ _ \/ __/ __| | |/ _ \ / _` |
| |_) | | | (_) | (_|  __/\__ \__ \ | | (_) | (_| |
| .__/|_|  \___/ \___\___||___/___/ |_|\___/ \__, |
|_|                                          |___/
*/
1                                          Altair SLC          13:33 Monday, March 23, 2026

NOTE: Copyright 2002-2025 World Programming, an Altair Company
NOTE: Altair SLC 2026 (05.26.01.00.000758)
      Licensed to Roger DeAngelis
NOTE: This session is executing on the X64_WIN11PRO platform and is running in 64 bit mode

NOTE: AUTOEXEC processing beginning; file is C:\wpsoto\autoexec.sas
NOTE: AUTOEXEC source line
1       +  ï»¿ods _all_ close;
           ^
ERROR: Expected a statement keyword : found "?"
NOTE: Library workx assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\wpswrkx

NOTE: Library slchelp assigned as follows:
      Engine:        WPD
      Physical Name: C:\Progra~1\Altair\SLC\2026\sashelp

NOTE: Library worksas assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\worksas

NOTE: Library workwpd assigned as follows:
      Engine:        WPD
      Physical Name: d:\workwpd


LOG:  13:33:20
NOTE: 1 record was written to file PRINT

NOTE: The data step took :
      real time : 0.016
      cpu time  : 0.000


NOTE: AUTOEXEC processing completed

1         options set=RHOME "C:\Progra~1\R\R-4.5.2\bin\r";
2
3         proc r;
NOTE: Using R version 4.5.2 (2025-10-31 ucrt) from C:\Program Files\R\R-4.5.2
4         export data=workx.iris r=iris;
NOTE: Creating R data frame 'iris' from data set 'WORKX.iris'

5         submit;
6
7         # Make Species a factor FIRST (critical fix)
8
9         iris$Species <- factor(iris$Species, levels=c("setosa","versicolor","virginica"))
10
11        # STRATIFIED SPLIT - BALANCED CLASSES
12        split_by_class <- function(df, train_prop=0.75) {
13          train_idx <- c()
14          for(sp in levels(df$Species)) {
15            idx <- which(df$Species == sp)
16            n_sp <- length(idx)
17            n_train_sp <- floor(train_prop * n_sp)
18            train_idx <- c(train_idx, sample(idx, n_train_sp))
19          }
20          return(train_idx)
21        }
22
23        train_idx <- split_by_class(iris, 0.75)
24        train <- iris[train_idx,]
25        test <- iris[-train_idx,]  # Now works!
26
27        cat("Train class balance:\n")
28        print(table(train$Species))
29        cat("Test class balance:\n")
30        print(table(test$Species))
31
32        # Rest of your PNN code runs perfectly now...
33        X_train <- as.matrix(train[,1:4])
34        y_train <- as.numeric(train$Species) - 1
35        X_test <- as.matrix(test[,1:4])
36        y_test <- as.numeric(test$Species) - 1
37        K <- 3
38        species_names <- c("setosa","versicolor","virginica")
39
40        # Your PNN function (unchanged - perfect)
41        pnn_gaussian <- function(X_train, y_train, X_test, sigma=0.3) {
42          n_test <- nrow(X_test)
43          probs <- matrix(0.001, n_test, K)
44          for(i in 1:n_test) {
45            x <- X_test[i,]
46            for(k in 1:K) {
47              class_idx <- which(y_train == (k-1))
48              if(length(class_idx) == 0) next
49              X_k <- X_train[class_idx,]
50              dists_sq <- rowSums((X_k - x)^2)
51              kernel <- exp(-dists_sq / (2 * sigma^2))
52              probs[i,k] <- mean(kernel)
53            }
54            probs[i,] <- probs[i,] / sum(probs[i,])
55          }
56          pred_classes <- (apply(probs, 1, which.max) - 1)
57          return(list(pred=pred_classes, probs=probs))
58        }
59
60        sigma <- 0.25
61        result <- pnn_gaussian(X_train, y_train, X_test, sigma)
62        pred_classes <- result$pred
63
64        accuracy <- mean(pred_classes == y_test)
65        cat(sprintf("\n FIXED PNN Accuracy: %.1f%% (%d/%d)\n",
66              accuracy*100, sum(pred_classes==y_test), length(y_test)))
67
68        # CONFUSION MATRIX
69        cat("\n CONFUSION MATRIX:\n")
70        true_f <- factor(y_test, levels=0:2, labels=species_names)
71        pred_f <- factor(pred_classes, levels=0:2, labels=species_names)
72        print(table(True=true_f, Predicted=pred_f))
73
74        # Base confusion matrix table
75        cm_table <- table(True=true_f, Predicted=pred_f)
76
77        # Convert to proper dataframe with row/column names
78        confusion_df <- as.data.frame(as.table(cm_table))
79        colnames(confusion_df) <- c("True_Species", "Pred_Species", "Count")
80
81        confusion_df
82
83        # TOP 5 PREDICTIONS
84        cat("\n TOP 5:\n")
85        for(i in 1:5) {
86          true_sp <- species_names[y_test[i]+1]
87          pred_sp <- species_names[pred_classes[i]+1]
88          maxp <- round(max(result$probs[i,])*100, 1)
89          cat(sprintf(" %d: %s > %s (%.1f%%)\n", i, true_sp, pred_sp, maxp))
90        }
91
92        # NEW FLOWER
93        new_flower <- matrix(c(6.5,3.0,5.5,2.0), nrow=1)
94        new_res <- pnn_gaussian(X_train, y_train, new_flower, sigma)
95        cat(sprintf("\n NEW %s (%.1f%%)\n",
96              species_names[new_res$pred[1]+1], max(new_res$probs[1,])*100))
97
98        endsubmit;

NOTE: Submitting statements to R:

>
> # Make Species a factor FIRST (critical fix)
>
> iris$Species <- factor(iris$Species, levels=c("setosa","versicolor","virginica"))
>
> # STRATIFIED SPLIT - BALANCED CLASSES
> split_by_class <- function(df, train_prop=0.75) {
+   train_idx <- c()
+   for(sp in levels(df$Species)) {
+     idx <- which(df$Species == sp)
+     n_sp <- length(idx)
+     n_train_sp <- floor(train_prop * n_sp)
+     train_idx <- c(train_idx, sample(idx, n_train_sp))
+   }
+   return(train_idx)
+ }
>
> train_idx <- split_by_class(iris, 0.75)
> train <- iris[train_idx,]
> test <- iris[-train_idx,]  # Now works!
>
> cat("Train class balance:\n")
> print(table(train$Species))
> cat("Test class balance:\n")
> print(table(test$Species))
>
> # Rest of your PNN code runs perfectly now...
> X_train <- as.matrix(train[,1:4])
> y_train <- as.numeric(train$Species) - 1
> X_test <- as.matrix(test[,1:4])
> y_test <- as.numeric(test$Species) - 1
> K <- 3
> species_names <- c("setosa","versicolor","virginica")
>
> # Your PNN function (unchanged - perfect)
> pnn_gaussian <- function(X_train, y_train, X_test, sigma=0.3) {
+   n_test <- nrow(X_test)
+   probs <- matrix(0.001, n_test, K)
+   for(i in 1:n_test) {
+     x <- X_test[i,]
+     for(k in 1:K) {
+       class_idx <- which(y_train == (k-1))
+       if(length(class_idx) == 0) next
+       X_k <- X_train[class_idx,]
+       dists_sq <- rowSums((X_k - x)^2)
+       kernel <- exp(-dists_sq / (2 * sigma^2))
+       probs[i,k] <- mean(kernel)
+     }
+     probs[i,] <- probs[i,] / sum(probs[i,])
+   }
+   pred_classes <- (apply(probs, 1, which.max) - 1)
+   return(list(pred=pred_classes, probs=probs))
+ }
>
> sigma <- 0.25
> result <- pnn_gaussian(X_train, y_train, X_test, sigma)
> pred_classes <- result$pred
>
> accuracy <- mean(pred_classes == y_test)
> cat(sprintf("\n FIXED PNN Accuracy: %.1f%% (%d/%d)\n",
+       accuracy*100, sum(pred_classes==y_test), length(y_test)))
>
> # CONFUSION MATRIX
> cat("\n CONFUSION MATRIX:\n")
> true_f <- factor(y_test, levels=0:2, labels=species_names)
> pred_f <- factor(pred_classes, levels=0:2, labels=species_names)
> print(table(True=true_f, Predicted=pred_f))
>
> # Base confusion matrix table
> cm_table <- table(True=true_f, Predicted=pred_f)
>
> # Convert to proper dataframe with row/column names
> confusion_df <- as.data.frame(as.table(cm_table))
> colnames(confusion_df) <- c("True_Species", "Pred_Species", "Count")
>
> confusion_df
>
> # TOP 5 PREDICTIONS
> cat("\n TOP 5:\n")
> for(i in 1:5) {
+   true_sp <- species_names[y_test[i]+1]
+   pred_sp <- species_names[pred_classes[i]+1]
+   maxp <- round(max(result$probs[i,])*100, 1)
+   cat(sprintf(" %d: %s > %s (%.1f%%)\n", i, true_sp, pred_sp, maxp))
+ }
>
> # NEW FLOWER
> new_flower <- matrix(c(6.5,3.0,5.5,2.0), nrow=1)
> new_res <- pnn_gaussian(X_train, y_train, new_flower, sigma)
> cat(sprintf("\n NEW %s (%.1f%%)\n",
+       species_names[new_res$pred[1]+1], max(new_res$probs[1,])*100))
>

NOTE: Processing of R statements complete

99        import r=confusion_df data=workx.confusion_df;
NOTE: Creating data set 'WORKX.confusion_df' from R data frame 'confusion_df'
NOTE: Column names modified during import of 'confusion_df'
NOTE: Data set "WORKX.confusion_df" has 9 observation(s) and 3 variable(s)

100       import r=test data=workx.test;
NOTE: Creating data set 'WORKX.test' from R data frame 'test'
NOTE: Column names modified during import of 'test'
NOTE: Data set "WORKX.test" has 39 observation(s) and 5 variable(s)

101       import r=train data=workx.train;
NOTE: Creating data set 'WORKX.train' from R data frame 'train'
NOTE: Column names modified during import of 'train'
NOTE: Data set "WORKX.train" has 111 observation(s) and 5 variable(s)

102       run;
NOTE: Procedure r step took :
      real time : 0.475
      cpu time  : 0.078


103
104
ERROR: Error printed on page 1

NOTE: Submitted statements took :
      real time : 0.554
      cpu time  : 0.156


/*              _
  ___ _ __   __| |
 / _ \ `_ \ / _` |
|  __/ | | | (_| |
 \___|_| |_|\__,_|

*/
