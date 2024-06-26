
#include "tensor.h"
#include "grad.h"

#define NO_GRAD false
#define HAS_GRAD true

class MLP {
	public:
		MLP(size_t in, size_t out);
		~MLP();
		
		Tensor forward(Tensor x);
		Tensor backward(Tensor x);
	
		Tensor weight;
		Tensor bias;
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
	size_t bias_sizes[2] = {n_out, 1};
	bias.self_init(bias_sizes, 2);
	bias.fill(0);
}


MLP::~MLP() {}


Tensor MLP::forward(Tensor x) {
	// should probably add context-storing functionality here
	Tensor c = weight ^ x;
//	c = c + bias;
//	c = c.tanh();
	return c;

}

class Network {
	public:
	Network() : layer1(5,6), layer2(6,6), layer3(6,5) {}
	
	Tensor forward(Tensor x) {
		x = layer1.forward(x);
		x = layer2.forward(x);
		x = layer3.forward(x);
		x = x.softmax();
		return x;
	}

	MLP layer1;
	MLP layer2;
	MLP layer3;

};

	// takes as input a (C, 1) tensor
	float cross_entropy(Tensor x, Tensor target) {
		size_t classes = x.get_shape()[0];
		float sum = 0;
		for(int i = 0; i < classes; i++) {
			sum += target[i] * -log2(x[i]); 
		}
		
		return sum;

	}
	
	Tensor cross_entropy_grad(Tensor x, Tensor target) {
		int idx;
		size_t* shape = x.get_shape(); 
		for(int i = 0; i < shape[1]; i++) {
			if(target[i] == 1) idx = i;
		}

		Tensor grad(shape, 2, NO_GRAD);					
		grad.fill(0);
		grad[idx] = -1/(x[idx]);
		return grad;

	} 
	
	void propagate(Tensor t, Tensor dL) {
		if(!t.backward) return;
		vector<Tensor> parents = t.backward->input_ctx;
		vector<Tensor> parent_grads = t.backward->compute_grads(dL);

		for(int i = 0; i < parents.size(); i++) {
			if(!parents[i].check_if_grad()) continue;
			parents[i].accumulate_grad(parent_grads[i]);
			propagate(parents[i], parent_grads[i]);
		}

	}

	void backward(Tensor x, Tensor target) {
		// first compute derivative wrt loss
		Tensor dL = cross_entropy_grad(x, target);
		propagate(x, dL);
	}

		

void traverse(Tensor x, float scaling_factor) {
	if(!x.backward) return;
	vector<Tensor> parents = x.backward->input_ctx;
	for(int i = 0; i < parents.size(); i++) {
		if(parents[i].get_grad()) parents[i].get_grad()->muleq(scaling_factor);
		traverse(parents[i], scaling_factor);
	}
}


void SGD(Network net, vector< pair<Tensor, Tensor> > batches) {
	size_t n = batches.size();
	for(int i = 0; i < n; i++) {
		Tensor x = batches[i].first;
		Tensor target = batches[i].second;
		Tensor res = net.forward(x);
		float loss_val = cross_entropy(res, target);	
		backward(res, target);
	}
	
//	traverse(res, (1/n));	
	


//	return res;
}

int main() {
	size_t shape[2] = {5, 1};
	Tensor input(shape, 2, false);
	input.fill(1);

	Tensor target(shape, 2, false);
	target.fill(0);
	target[0] = 1;


	Tensor input2(shape, 2, false);
	input2.fill(1);

	Tensor target2(shape, 2, false);
	target2.fill(0);
	target2[1] = 1;

	/*
	Tensor res = net.forward(input);
	
	backward(res, target);

//	SGD(net, batches);
*/	

	MLP layer1(5,5);	
	MLP layer2(5,5);
	MLP layer3(5,5);
	MLP layer4(5,5);
	MLP layer5(5,5);
	MLP layer6(5,5);
	MLP layer7(5,5);
	MLP layer8(5,5);
	MLP layer9(5,5);
	

	Tensor res = layer9.forward(layer8.forward(layer7.forward(layer6.forward(layer5.forward(layer4.forward(layer3.forward(layer2.forward(layer1.forward(input)))))))));
	backward(res, target);

	return 0;


}


