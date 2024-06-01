
#include "tensor.h"
#include "grad.h"


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

};


MLP::MLP(size_t in, size_t out) {
	n_in = in;
	n_out = out;
	size_t sizes[2] = {out, in};
	weight.self_init(sizes, 2);
	weight.xavier_init();

}


MLP::~MLP() {}


Tensor MLP::forward(Tensor x) {
	// should probably add context-storing functionality here
	Tensor c = weight ^ x;
	c = c.tanh();
//	c.tanh();
	return c;

}

Tensor MLP::backward(Tensor x) {
	// for now backward passes are not defined
	return x;

}


class Loss {
	public:
		Loss() {}
		// takes as input a (C, 1) tensor
		float cross_entropy(Tensor x_, Tensor target_) {
			x = x_;
			target = target_;
			size_t classes = x.get_shape()[0];
			float sum = 0;
			for(int i = 0; i < classes; i++) {
				sum += target[i] * -log2(x[i]); 
			}
			
			loss_val = sum;
			return sum;

		}
		
		Tensor cross_entropy_grad() {
			int idx;
			size_t* shape = x.get_shape(); 

			for(int i = 0; i < shape[1]; i++) {
				if(target[i] == 1) idx = i;
			}

			Tensor grad(shape, 2, false);					
			grad.fill(0);
			grad[idx] = -1/(x[idx]);
			return grad;

		} 
		
		void propagate(Tensor t, Tensor dL) {
			vector<Tensor> parent_grads = t.backward->compute_grads(dL);
			vector<Tensor> parents = t.backward->input_ctx; 
//			parents[1].dump();
		/*	for(int i = 0; i < parents.size(); i++) {
				parents[i].populate_grad(parent_grads[i]);
				propagate(parents[i], parent_grads[i]);
			}
		*/}

		void backward() {
			// first compute derivative wrt loss
			Tensor dL = cross_entropy_grad();
			propagate(x, dL);
		}

		
	private:
		float loss_val;
		Tensor x;
		Tensor target;
		

};

int main() {
	Loss loss;
	size_t shape[2] = {5, 1};
	Tensor input(shape, 2, false);
	input.fill(1);
	MLP layer1(5, 6);
	MLP layer2(6, 6);
	MLP final_layer(6, 5);
	Tensor res = layer1.forward(input);	
	res = layer2.forward(res);
	res = final_layer.forward(res);
	res = res.softmax();
	
	Tensor target(shape, 2, false);
	target.fill(0);
	target[0] = 1;


	cout << loss.cross_entropy(res, target) << endl;
		
	loss.backward();
	

//	cout << res.get_grad()->get_shape()[0] << endl;
	//res.dump();

	/*
	size_t shape1[2] = {4,1};
	size_t shape2[2] = {3,4};
	Tensor a(shape1, 2, true);
	a.fill(1);
	Tensor W(shape2, 2, true); 
	W.fill(2);
	cout << W.get_grad() << endl;
	Tensor out = W ^ a;

	size_t shape3[2] = {3,1};
	Tensor ograd(shape3, 2, false);
	ograd.fill(2);
	*/
	return 0;


}


