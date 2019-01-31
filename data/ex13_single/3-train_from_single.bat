del output_single\*.* /Q
..\opencv_traincascade -data output_single -vec pos_single.vec -bg neg\neg_train.txt -numPos 100 -numNeg 10 -numStages 10 -w 40 -h 40
