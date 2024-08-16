#Creates classification/labelling of transformer output using a linear layer followed by softmax
#if from 0 to 1 scale, a label has a rating > 0.5, consider it to match that label
#incorrect if both neutral & a negative label > 0.5