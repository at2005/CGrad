
#ifndef GRAD_H
#define GRAD_H

#include "tensor.h"

typedef enum {ADD, SUB, MMUL, SIG} diff_op_t;

class AutoDiffFunc {
	public:
		AutoDiffFunc(diff_op_t op_type, vector<Tensor> inputs) {
			op = op_type;	
			for(int i = 0; i < inputs.size(); i++) {
				input_ctx.push_back(inputs[i]);
			}
			//input_ctx = inputs;

		}

		vector<Tensor> compute_grads(Tensor* output_grad) {
			switch(op) {
				case MMUL:
					// inputs must be in order W, x
					Tensor weights = input_ctx[0];
					Tensor x = input_ctx[1];
					Tensor* dW = x.make_copy();
					dW->inplace_transpose(); 
					Tensor* dx = weights.make_copy();
					dx->inplace_transpose();
					
					Tensor* acc_grad_W = (*output_grad ^ *dW).make_copy();
					Tensor* acc_grad_x = (*dx ^ *output_grad).make_copy();
					
					acc_grad_x->dump();
					acc_grad_W->dump();
					//delete dx;
					//delete dW;


					vector<Tensor> grads;
					grads.push_back(*acc_grad_W);
					grads.push_back(*acc_grad_x);
					return grads;

			}	
		}


	diff_op_t op;
	vector<Tensor> input_ctx;

};

#endif
