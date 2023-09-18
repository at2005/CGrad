
#include <iostream>

using namespace std;

class Vector {
public:

	Vector(unsigned int* shape, unsigned int dim) {
		this->shape = shape;
		this->dim = dim;
		int num_elements = 1;
		for(int i = 0; i < dim; i++) {
			num_elements *= shape[i];
		}

		this->num_el = num_elements;
		
		this->data_buffer = (double*)malloc(sizeof(double) * num_elements);
	}


	unsigned int* shape;
	unsigned int dim;
	double* data_buffer;
	unsigned int num_el;

	
	Vector operator+(Vector& b) {
		unsigned int* shape_vec_copy = (unsigned int*)malloc(sizeof(unsigned int) * this->dim);
		memcpy(shape_vec_copy, this->shape, this->dim);

		Vector c = Vector(this->shape, this->dim); 
		for(int i = 0; i < this->num_el; i++) {
			c.data_buffer[i] = this->data_buffer[i] + b.data_buffer[i];
		}
		
	}

};


double dot_prod(Vector a, Vector b) {
	double total = 0;
	for(int i = 0; i < a.shape; i++) {
		total += a[i] * b[i];					
	}


}

