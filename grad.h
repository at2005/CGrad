
#ifndef GRAD_H
#define GRAD_H

#include "tensor.h"

typedef enum {ADD, SUB, MMUL, TANH, SOFTMAX} diff_op_t;

class AutoDiffFunc {
	public:
		AutoDiffFunc(diff_op_t op_type, vector<Tensor> inputs) {
			op = op_type;	
			for(int i = 0; i < inputs.size(); i++) {
				input_ctx.push_back(inputs[i]);
			}
			//input_ctx = inputs;

		}

		vector<Tensor> compute_grads(Tensor output_grad) {
			switch(op) {
				case MMUL:
					// inputs must be in order W, x
					Tensor weights = input_ctx[0];
					Tensor x = input_ctx[1];
					Tensor* dW = x.make_copy();
					dW->inplace_transpose(); 
					Tensor* dx = weights.make_copy();
					dx->inplace_transpose();
					
					Tensor* acc_grad_W = (output_grad ^ *dW).make_copy();
					Tensor* acc_grad_x = (*dx ^ output_grad).make_copy();
					
					grads.push_back(*acc_grad_W);
					grads.push_back(*acc_grad_x);
					break;

				case TANH:
					Tensor tanh_grad(input_ctx[0]);
					tanh_grad.fill(1);
					tanh_grad = tanh_grad - (input_ctx[0] * input_ctx[0]);
					grads.push_back(tanh_grad * output_grad);
					break;

				case SOFTMAX:
					Tensor s = input_ctx[0];
					size_t n = s.shape[0];
					size_t shape_grad_mat[2] = {n,n};
					Tensor softmax_grad_matrix(shape_grad_mat, 2, false);
					for(int i = 0; i < n; i++) {
						for(int j = 0; j < n; j++) {
							if(i == j) softmax_grad_matrix[i*n + j] = (1-s[i]) * s[i];
							else softmax_grad_matrix[i*n + j] = -s[i] * s[j];
						}
					}

					grads.push_back(softmax_grad_matrix ^ output_grad);
					break;

			}
			
			
			return grads;

		}


	vector<Tensor> grads;
	diff_op_t op;
	vector<Tensor> input_ctx;

};

#endif
