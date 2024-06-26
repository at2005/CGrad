
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
using namespace std;

class AutoDiffFunc;


class Tensor {
	public:
		Tensor(size_t* in_shape, size_t in_dim, bool has_grad=true);
//		Tensor(const Tensor& t);
		Tensor();
		~Tensor() ;
		
		void self_init(size_t* in_shape, size_t in_dim, bool has_grad=true);

		float* get_mem();

		size_t* get_shape();

		size_t get_dim();

		size_t get_size();
		
		float& operator[](size_t i);

		Tensor operator+(Tensor b); 

//		Tensor operator=(Tensor b); 
		
		Tensor operator-(Tensor b);

		Tensor operator*(float scalar);
		
		// element-wise hadamard product
		Tensor operator*(Tensor b);
		void muleq(float c);

		//Tensor operator/(float scalar);
		
		// matmul operation
		Tensor operator^(Tensor b);
	
		void pretty_shape();
		void dump();
		
		Tensor* get_grad();
		void accumulate_grad(Tensor grad_value);
		bool check_if_grad() {
			return calc_grad;	
		}

		void fill(float value);
		
		void xavier_init();

		Tensor tsqrt();	

		Tensor texp();

		Tensor tanh();
		
		Tensor softmax();
		
		void inplace_transpose();
		
		Tensor* make_copy();

		AutoDiffFunc* backward;
			

	private:
		float* mem_block;
		size_t* shape;
		size_t dim;
		size_t size;
		vector<Tensor*> children;	
		Tensor* grad;
		bool calc_grad;		
		size_t cumprod(size_t* arr, size_t len);
		void create_prod_arr(size_t* shape_arr, size_t arr_size, int* output_arr, size_t curr_index);
		


};


#endif


