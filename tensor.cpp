
#include "tensor.h"

Tensor::Tensor(size_t* in_shape, size_t in_dim) {
	this->self_init(in_shape, in_dim);

}

void Tensor::self_init(size_t* in_shape, size_t in_dim) {

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
}


Tensor::Tensor() {}

Tensor::~Tensor() {}


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

float& Tensor::operator[](size_t i) {
	return mem_block[i];

}


Tensor Tensor::operator+(Tensor b) {
	Tensor c = *(new Tensor(this->get_shape(), this->get_dim()));
	for(int i = 0; i < this->get_size();i++) {
		c[i] = this->mem_block[i] + b[i];

	}

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

	for(int i = 0; i < a_num_rows; i++) {
		for(int j = 0; j < b_num_cols; j++) {
			float total = 0.0;
			for(int k = 0; k < a_num_cols; k++) {
				total += this->mem_block[i*a_num_cols + k] * b[j + k*b_num_cols];
			}

			c[i*b_num_cols + j] = total;
		}
	}


	return c;
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

void Tensor::tsqrt() {
	for(int i = 0; i < size; i++) mem_block[i] = sqrt(mem_block[i]);

}

void Tensor::texp() {
	for(int i = 0; i < size; i++) mem_block[i] = exp(mem_block[i]);

}

float tanh_scalar(float z) {
	return (exp(z) - exp(-z)) / (exp(z) + exp(-z)); 

}

void Tensor::tanh() {
	for(int i = 0; i < size; i++) mem_block[i] = tanh_scalar(mem_block[i]);
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







// testing
/*
int main() {
	
	size_t shape[2] = {8,4};
	
	Tensor a(shape, 2);		
	a.fill(2.5);	


	size_t test_shape[2] = {4, 8};
	Tensor test(test_shape, 2);
	test.fill(2.5);

//	for(int i = 0; i < 16; i++) {a[i] = (float)i; b[i] = (float)i;}

	Tensor c = a ^ test;
	c.dump();
*c = c ^ a;		
	c = c * -1.43;
	c = c ^ b;
	c = a + c;
	c = c * -1.5;
	c.dump();
	c.fill(2.0);
	
//	c.dump();
	return 0;

}
*/
