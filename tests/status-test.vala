// vapidirs: vapi
// modules: tensorflow

using TensorFlow;

public int main (string[] args) {
	Test.init (ref args);

	Test.add_func ("/status", () => {
		var status = new Status ();
		Code[] codes = {
			Code.OK,
			Code.CANCELLED,
			Code.UNKNOWN,
			Code.INVALID_ARGUMENT,
			Code.DEADLINE_EXCEEDED,
			Code.NOT_FOUND,
			Code.ALREADY_EXISTS,
			Code.PERMISSION_DENIED,
			Code.UNAUTHENTICATED,
			Code.RESOURCE_EXHAUSTED,
			Code.FAILED_PRECONDITION,
			Code.ABORTED,
			Code.OUT_OF_RANGE,
			Code.UNIMPLEMENTED,
			Code.INTERNAL,
			Code.UNAVAILABLE,
			Code.DATA_LOSS,
		};
		var message = "Test status message";
		foreach (var code in codes) {
			status.set (code, message);
			assert (status.code () == code);
			assert (status.message () == message);
		}
	});

	return Test.run ();
}
