
#include <cmath>

vector<pair<Tensor> > make_sample_batches(size_t batch_size, size_t* shape, size_t dim) {
	vector<pair<Tensor> > batches;
	for(int i = 0; i < batch_size; i++) {
		Tensor x(shape, dim, false);
		x.xavier_init();
		Tensor target = x.sin();
		batches.push_back(make_pair(x, target));
	}

	return batches;
}
