
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
		
		Tensor create_prod_arr(size_t* shape_arr, size_t arr_size, int* output_arr, size_t curr_index);

};





int main() {
	
	size_t shape[2] = {4,4};

	Tensor a(shape, 2);		
	Tensor b(shape, 2);


	for(int i = 0; i < 16; i++) {a[i] = (float)i; b[i] = (float)i;}

	Tensor c = a ^ b;
	c.dump();	
	c = c ^ a;		
	c = c * -1.43;
	c = c ^ b;
	c = a + c;
	c = c * -1.5;
	c.dump();
	c.fill(2.0);
	c.dump();
	return 0;

}

