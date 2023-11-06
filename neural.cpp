
#include "tensor.h"

class MLP {
	public:
		MLP(size_t in, size_t out);
		~MLP();
		
		Tensor forward(Tensor x);
		Tensor backward(Tensor x);
	
		Tensor weight;
	private:
		
		size_t n_in;
		size_t n_out;
		// weight matrix

};


MLP::MLP(size_t in, size_t out) {
	n_in = in;
	n_out = out;
	size_t sizes[2] = {in, out};
	weight.self_init(sizes, 2);
	weight.xavier_init();

}


MLP::~MLP() {

}


Tensor MLP::forward(Tensor x) {
	// should probably add context-storing functionality here
	Tensor c = weight ^ x;
	return c;

}

Tensor MLP::backward(Tensor x) {
	// for now backward passes are not defined
	return x;

}

int main() {
	size_t shape[2] = {64, 1};
	Tensor input(shape, 2);
	input.fill(1);
	MLP layer1(32, 64);
	Tensor res = layer1.forward(input);	
	res.dump();
	
	return 0;



}


