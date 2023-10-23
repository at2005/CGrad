
#include <vector>
#include <iostream>
#include <iomanip>
using namespace std;



class Tensor {
	public:
		Tensor(size_t* in_shape, size_t in_dim);

		~Tensor() ;
		
		float* get_mem();
		size_t* get_shape();

		size_t get_dim();


		size_t get_size();
		
		float& operator[](size_t i);

	
		
		Tensor operator+(Tensor b); 
		

		Tensor operator*(float scalar);

		Tensor operator^(Tensor b);


		void dump() ;


		void fill(float value) ;


	private:
		float* mem_block;
		size_t* shape;
		size_t dim;
		size_t size;

		size_t cumprod(size_t* arr, size_t len);
		
		void create_prod_arr(size_t* shape_arr, size_t arr_size, int* output_arr, size_t curr_index);

};





