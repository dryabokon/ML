del output_multip\*.* /Q
opencv_traincascade -data output_multip -vec pos_multip.vec -bg neg\neg_train.txt -numPos 4 -numNeg 10 -numStages 10 -w 28 -h 28
