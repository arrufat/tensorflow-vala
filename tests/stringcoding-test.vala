// vapidirs: vapi
// modules: tensorflow

using TensorFlow;

public int main (string[] args) {
	Test.init (ref args);

	Test.add_func ("/string_encode", () => {
		var status = new Status ();
		var test_string = "A long test string with 1 number";
		var dec = test_string;
		var enc = "it does not matter what's in here...";
		size_t dst_length;
		print ("%s\n", test_string);
		var enc_size = string_encode (dec, dec.length + 1, enc, string_encoded_size(dec.length + 1), status);
		stdout.printf ("src: %s (%lu)\n", dec, dec.length);
		stdout.printf ("dst: %s (%lu)\n", enc, enc_size);
		enc_size = string_decode (enc, enc_size, out dec, out dst_length, status);
		stdout.printf ("src: %s (%lu)\n", enc, enc_size);
		stdout.printf ("dst: %s (%lu)\n", dec, dec.length);
		assert (dec == test_string);
	});

	return Test.run ();
}
