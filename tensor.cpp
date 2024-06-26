
#include "tensor.h"
#include "grad.h"

Tensor::Tensor(size_t* in_shape, size_t in_dim, bool has_grad) {
	this->self_init(in_shape, in_dim, has_grad);

}

void Tensor::self_init(size_t* in_shape, size_t in_dim, bool has_grad) {

	// total number of elements given by a cumulative product of shape array over each dim
	size_t total_el = cumprod(in_shape, in_dim);

	// contiguous block of memory reserved for tensor values
	float* tensor_mem = (float*)malloc(sizeof(float) * total_el);

	// allocate memory for dimension array
	this->shape = (size_t*)malloc(sizeof(size_t) * in_dim);
	// copy shape array to memory
	memcpy(shape, in_shape, in_dim * sizeof(size_t));

	this->dim = in_dim;
	this->size = total_el;
	this->mem_block = tensor_mem;
	this->calc_grad = has_grad;

	if(has_grad) {
		this->grad = new Tensor(this->shape, this->dim, false);
		this->grad->self_init(this->shape, dim, false);
		this->grad->fill(0);

	}
	


}


Tensor::Tensor() {}

Tensor::~Tensor() {
//	free(mem_block);
//	free(shape);

}


/*
// copy constructor
Tensor::Tensor(const Tensor& t) {
	//t.dump();
	self_init(t.shape, t.dim, t.calc_grad);

}
*/

void Tensor::muleq(float c) {
	for(int i = 0; i < size; i++) mem_block[i] *= c;
}

float* Tensor::get_mem() {
	return mem_block;
}

size_t* Tensor::get_shape() {
	return shape;
}

size_t Tensor::get_dim() {
	return dim;

}

size_t Tensor::get_size() {
	return size;
}

Tensor* Tensor::get_grad() {
	return (this->grad);	
}

void Tensor::accumulate_grad(Tensor grad_value) {
	if(!this->grad) {
		this->grad = new Tensor(this->shape, this->dim, false);
		this->grad->fill(0);
	}
	*(this->grad) = *(this->grad) + grad_value;
}

float& Tensor::operator[](size_t i) {
	return mem_block[i];

}

/*
Tensor Tensor::operator=(Tensor b) {
	this->shape = b.get_shape();
	this->dim = b.get_dim();
	this->mem_block = b.get_mem();
	this->grad = b.get_grad();
	return *this;
}
*/


Tensor Tensor::operator+(Tensor b) {
	Tensor c = *(new Tensor(this->get_shape(), this->get_dim()));
	for(int i = 0; i < this->get_size();i++) {
		c[i] = this->mem_block[i] + b[i];
	
	}
	
	vector<Tensor> inputs;
	inputs.push_back(*this);
	inputs.push_back(b);
	c.backward = new AutoDiffFunc(ADD, inputs);
		
	return c;

}

Tensor Tensor::operator-(Tensor b) {
	Tensor c = *(new Tensor(this->get_shape(), this->get_dim()));
	for(int i = 0; i < this->get_size();i++) {
		c[i] = this->mem_block[i] - b[i];

	}

	return c;

}

Tensor Tensor::operator*(float scalar) {
	Tensor c = *(new Tensor(this->get_shape(), this->get_dim()));
	for(int i = 0; i < this->get_size();i++) {
		c[i] = scalar * this->mem_block[i];

	}

	return c;

}


Tensor Tensor::operator*(Tensor b) {
	Tensor c = *(new Tensor(this->get_shape(), this->get_dim()));
	
	for(int i = 0; i < this->get_size();i++) {
		c[i] = b.mem_block[i] * this->mem_block[i];

	}

	return c;


}

// matmul
Tensor Tensor::operator^(Tensor b) {

	if(this->get_dim() < 2 or b.get_dim() < 2) {
		cerr << "error\n";
		exit(1);

	}

	size_t* a_shape = this->get_shape();
	size_t* b_shape = b.get_shape();

	size_t a_num_rows = a_shape[this->get_dim() - 2];
	size_t a_num_cols = a_shape[this->get_dim() - 1];
	size_t b_num_rows = b_shape[b.get_dim() - 2];
	size_t b_num_cols = b_shape[b.get_dim() - 1];
	
	size_t shape[2] = {a_num_rows, b_num_cols};

	Tensor c  = *(new Tensor(shape, this->get_dim()));
	// inefficient matmul
	for(int i = 0; i < a_num_rows; i++) {
		for(int j = 0; j < b_num_cols; j++) {
			float total = 0.0;
			for(int k = 0; k < a_num_cols; k++) {
				total += this->mem_block[i*a_num_cols + k] * b[j + k*b_num_cols];
			}

			c[i*b_num_cols + j] = total;
		}
	}
	
	std::vector<Tensor> inputs;
	inputs.push_back(*this);
	inputs.push_back(b);

	c.backward = new AutoDiffFunc(MMUL, inputs);	

	return c;
}


void Tensor::pretty_shape() {
	std::cout << "(";
	for(int i = 0; i < dim; i++) std::cout << shape[i] << ",";
	std::cout << ")\n";


}

void Tensor::dump() {
	float* buff = mem_block;
	size_t num_dim = dim;

	std::cout << std::fixed;
	 std::cout << std::setprecision(4);
	int prod_arr[num_dim];
	// here we create a product array where we store the cumulative product for the following elements of each element
	create_prod_arr(shape, num_dim, prod_arr, num_dim-1);

	int total_elements = prod_arr[0];

	for(int j = 0; j < total_elements; j++) {
		std::cout << buff[j] << " ";
		//printf("%f ", round(buff[j])); 
		for(int i = 0; i < num_dim; i++) {
			if((j+1) % prod_arr[i] == 0) {
				printf("\n");
			}

		}
	}

}


void Tensor::fill(float value) {
	for(int i = 0; i < size; i++) mem_block[i] = value;

}

void Tensor::xavier_init() {
	size_t in = this->shape[dim - 1];
	size_t out = this->shape[dim - 2];
		
	std::random_device rand_dev;
	std::mt19937 generate(rand_dev());
	std::normal_distribution<> gaussian_distr(0, 2.0/(in + out));
	
	for(int i = 0; i < this->size; i++) {
		mem_block[i] = gaussian_distr(generate);

	}

} 

Tensor Tensor::tsqrt() {
	Tensor new_tensor(*this);
	for(int i = 0; i < size; i++) new_tensor.mem_block[i] = sqrt(this->mem_block[i]);
	return new_tensor;

}

Tensor Tensor::texp() {
	Tensor new_tensor(*this);
	for(int i = 0; i < size; i++) new_tensor.mem_block[i] = exp(this->mem_block[i]);
	return new_tensor;

}

float tanh_scalar(float z) {
	return (exp(z) - exp(-z)) / (exp(z) + exp(-z)); 

}

Tensor Tensor::tanh() {
	Tensor new_tensor(*this);
	for(int i = 0; i < size; i++) new_tensor.mem_block[i] = tanh_scalar(this->mem_block[i]);
	vector<Tensor> ctx;
	ctx.push_back(new_tensor);
	new_tensor.backward = new AutoDiffFunc(TANH, ctx);
	return new_tensor;

}

Tensor Tensor::softmax() {
	Tensor new_tensor(*this);
	
	float running_total = 0;
	for(int i = 0; i < size; i++) {
		new_tensor.mem_block[i] = exp(this->mem_block[i]);
		running_total += new_tensor.mem_block[i];
	}

	for(int i = 0; i < size; i++) new_tensor.mem_block[i] /= running_total;

	vector<Tensor> ctx;
	ctx.push_back(new_tensor);
	new_tensor.backward = new AutoDiffFunc(SOFTMAX, ctx);
	return new_tensor;

}

// cumulative product function
size_t Tensor::cumprod(size_t* arr, size_t len) {
	size_t prod = 1;
	for(int i = 0; i < len; i++) prod *= arr[i];
	return prod;

}

void Tensor::create_prod_arr(size_t* shape_arr, size_t arr_size, int* output_arr, size_t curr_index) {
	if(curr_index == arr_size-1) {
		output_arr[arr_size - 1] = shape_arr[arr_size-1];
	}

	else {
		output_arr[curr_index] = output_arr[curr_index+1] * shape_arr[curr_index];
	}

	if(curr_index == 0) return;

	create_prod_arr(shape_arr, arr_size, output_arr, curr_index-1);
}


void Tensor::inplace_transpose() {
	// defined as swapping the last two dimensions
	if(dim < 2) cout << "Error. Tensor must have at least two dimensions to be transposed.\n";			
	size_t last_dim = shape[dim-1];
	shape[dim-1] = shape[dim-2];
	shape[dim-2] = last_dim;
	
}



Tensor* Tensor::make_copy() {
	Tensor* cpy_tensor = new Tensor(this->shape, this->dim, this-calc_grad);
	memcpy(cpy_tensor->get_mem(), this->mem_block, this->size*sizeof(float));
	return cpy_tensor;
}
