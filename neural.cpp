
#include "tensor.h"

class MLP {
	public:
		MLP(size_t in, size_t out);
		~MLP();
		
		Tensor forward(Tensor x);
		Tensor backward(Tensor x);
	
	private:
		size_t in_dim;
		size_t out_dim;
		
		// weight matrix
		Tensor weight;

};


MLP::MLP(size_t in, size_t out) {
	this->in_dim = in;
	this->out_dim = out;
	
	// height is out_dim, width is in_dim 	
	size_t weight_shape = {2,2};	
	this->weight = *(new Tensor(weight_shape, 2));

}


MLP::~MLP() {
	delete &(this->weight);

}


Tensor MLP::forward(Tensor x) {
	return weight ^ x;
}

Tensor MLP::backward(Tensor x) {
	// for now backward passes are not defined
	return x;

}


