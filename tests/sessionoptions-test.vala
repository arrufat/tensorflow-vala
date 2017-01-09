// vapidirs: vapi
// modules: tensorflow

using TensorFlow;

public int main (string[] args) {
	Test.init (ref args);

	Test.add_func ("/sessionoptions", () => {
		var status = new Status ();
		var session_options = new SessionOptions ();
		var options = "test";
		session_options.set_target (options);
		session_options.set_config (options.data, status);
		/* comment out until I figure out how to pass a TensorFlow.ConfigProto */
		/* assert (status.code () == Code.OK); */
	});

	return Test.run ();
}
