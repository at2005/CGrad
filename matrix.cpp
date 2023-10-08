
#include <vector>
#include <iostream>

using namespace std;



class Tensor {
	public:
		Tensor(uint64_t* in_shape, size_t in_dim) {
			size_t total_el = cumprod(in_shape, in_dim);
			float* tensor_mem = (float*)malloc(sizeof(float) * total_el);
				
			this->shape = in_shape;
			this->dim = in_dim;
			this->total_elements = total_el;
			this->mem_block = tensor_mem;
			
		}

		~Tensor() {
			free(this->mem_block);

		}
		
		float* get_mem() {
			return mem;
		}

		uint64_t* get_shape() {
			return shape;
		}

		size_t get_dim() {
			return dim;

		}

		size_t get_size() {
			return size;
		}
		
		float& operator[](size_t i) {
			return mem_block[i];

		}

	private:
		float* mem_block;
		uint64_t* shape;
		size_t dim;
		size_t size;

		// cumulative product function
		size_t cumprod(uint64_t* arr, size_t len) {
			size_t prod = 1;
			for(int i = 0; i < len; i++) prod *= arr[i];
			return prod;

		}

}

Tensor add_tensor(Tensor a, Tensor b) {
	Tensor c(a.get_shape(), a.get_dim());
	for(int i = 0; i < a.get_total_el(); i++) {
		c[i] = a[i] + b[i];	

	}
	
	return c;

}


float matmul(Tensor a, Tensor b) {
	if(a.dim < 2 or b.dim < 2) {
		cerr << "error\n";
		exit(1);

	}	
	
	size_t a_num_rows = a.shape[a.dim - 2];
	size_t a_num_cols = a.shape[a.dim - 1];
	size_t b_num_rows = b.shape[b.dim - 2];
	size_t b_num_cols = b.shape[b.dim - 1];
	
	for(int i = 0; i < a_num_rows; i++) {
		for(int j = 0; j < b_col_sz; j++) {
			float total = 0.0;
			for(int k = 0; k < a_col_sz; i++) {
				total += a[i*a_row_sz + k] * b[j + k*b_row_sz];
			}
		}
	}

}



