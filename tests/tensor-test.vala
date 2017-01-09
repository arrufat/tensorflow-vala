// vapidirs: vapi
// modules: tensorflow

using TensorFlow;

public int main (string[] args) {
	Test.init (ref args);

	Test.add_func ("/tensor", () => {
		DataType[] types = {
			DataType.FLOAT,
			DataType.DOUBLE,
			DataType.INT32,
			DataType.UINT8,
			DataType.INT16,
			DataType.INT8,
			DataType.STRING,
			DataType.COMPLEX64,
			DataType.COMPLEX,
			DataType.INT64,
			DataType.BOOL,
			DataType.QINT8,
			DataType.QUINT8,
			DataType.QINT32,
			DataType.BFLOAT16,
			DataType.QINT16,
			DataType.QUINT16,
			DataType.UINT16,
			DataType.COMPLEX128,
			DataType.HALF,
			DataType.RESOURCE,
		};
		long[] dims = {1, 2, 3, 4, 5, 6, 7, 8, 9};
		size_t byte_size = 256;
		foreach (var type in types) {
			var tensor = new Tensor (type, dims, byte_size);
			assert (tensor.type () == type);
		}
		var tensor = new Tensor (DataType.FLOAT, dims, byte_size);
		assert (tensor.byte_size () == byte_size);
		assert (tensor.num_dim() == dims.length);
		for (var i = 0; i < tensor.num_dim (); i++) {
			assert (dims[i] == i + 1);
		}
	});

	return Test.run ();
}
