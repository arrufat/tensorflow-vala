/* tensorflow.vapi
 * 
 * Copyright (C) 2017 Adrià Arrufat <adria.arrufat@protonmail.ch>
 * 
 * License: Apache 2.0
 * 
 * Author:
 *	Adrià Arrufat <adria.arrufat@protonmail.ch>
 */

/**
 * The main namespace containing all current functionality.
 */
[CCode (cprefix = "TF_", cheader_filename = "tensorflow/c_api.h")]
namespace TensorFlow {

	/**
	 * Describes the version information of the TensorFlow library using semantic versioning
	 * 
	 * @return version of the TensorFlow library as a string
	 */
	[CCode (cname = "TF_Version")]
	public unowned string version ();

	/**
	 * Holds the type for a scalar value.  E.g., one slot in a tensor.
	 * The enum values here are identical to corresponding values in types.proto.
	 */
	[CCode (cprefix = "TF_", has_type_id = false)]
	public enum DataType {
		FLOAT = 1,
		DOUBLE = 2,
		INT32 = 3,  // Int32 tensors are always in 'host' memory.
		UINT8 = 4,
		INT16 = 5,
		INT8 = 6,
		STRING = 7,
		COMPLEX64 = 8,  // Single-precision complex
		COMPLEX = 8,    // Old identifier kept for API backwards compatibility
		INT64 = 9,
		BOOL = 10,
		QINT8 = 11,     // Quantized int8
		QUINT8 = 12,    // Quantized uint8
		QINT32 = 13,    // Quantized int32
		BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
		QINT16 = 15,    // Quantized int16
		QUINT16 = 16,   // Quantized uint16
		UINT16 = 17,
		COMPLEX128 = 18,  // Double-precision complex
		HALF = 19,
		RESOURCE = 20,
	}

	/**
	 * Holds an error code.  The enum values here are identical to
	 * corresponding values in error_codes.proto.
	 */
	[CCode (cname = "TF_Code", cprefix = "TF_", has_type_id = false)]
	public enum Code {
		OK = 0,
		CANCELLED = 1,
		UNKNOWN = 2,
		INVALID_ARGUMENT = 3,
		DEADLINE_EXCEEDED = 4,
		NOT_FOUND = 5,
		ALREADY_EXISTS = 6,
		PERMISSION_DENIED = 7,
		UNAUTHENTICATED = 16,
		RESOURCE_EXHAUSTED = 8,
		FAILED_PRECONDITION = 9,
		ABORTED = 10,
		OUT_OF_RANGE = 11,
		UNIMPLEMENTED = 12,
		INTERNAL = 13,
		UNAVAILABLE = 14,
		DATA_LOSS = 15,
	}

	/**
	 * Holds error information. It either has an OK code, or
	 * else an error code with an associated error message.
	 */
	[CCode (cname = "TF_Status", free_function = "TF_DeleteStatus")]
	[Compact]
	public class Status {
		[CCode (cname = "TF_NewStatus")]
		public Status ();
		[CCode (cname = "TF_SetStatus")]
		public void set (Code code, string msg);
		[CCode (cname = "TF_GetCode")]
		public Code code ();
		[CCode (cname = "TF_Message")]
		public unowned string message ();
	}

	
	/**
	 * Holds a pointer to a block of data and its associated length.
	 *
	 * Typically the data consists of a serialized protocol buffer, but other data may also be held in a buffer
	 */
	[CCode (cname = "TF_Buffer", free_function = "TF_DeleteBuffer")]
	[Compact]
	public class Buffer {
		[CCode (cname = "data_deallocator")]
		public delegate void DataDeallocator (uint8[] data);
		[CCode (cname = "TF_NewBuffer")]
		public Buffer ();
		[CCode (cname = "TF_NewBufferFromString")]
		public Buffer.from_string (uint8[] proto);
		[CCode (cname = "TF_GetBuffer")]
		public Buffer @get ();
	}
	
	/**
	 * Holds a multi-dimensional array of elements of a single data type.
	 *
	 * For all types other than {@link DataType.STRING}, the data buffer stores elements in row major order.
	 * E.g. if data is treated as a vector of {@link DataType}:
	 * {{{
	 *     element 0:    index (0, ..., 0)
	 *     element 1:    index (0, ..., 1)
	 *     ...
	 * }}}
	 * The format for {@link DataType.STRING} tensors is:
	 * {{{
	 *     start_offset: array[uint64]
	 *     data:         byte[...]
	 * }}}
	 * The string length (as a varint), followed by the contents of the string is encoded at: {{{data[start_offset[i]]}}}
	 * {@link string_encode} and {@link string_decode} facilitate this encoding.
	 * @see string_encode
	 * @see string_decode
	 */
	[CCode (cname = "TF_Tensor", free_function = "TF_DeleteTensor")]
	[Compact]
	public class Tensor {
		
		/**
		 * Return a new tensor that holds the bytes data.
		 *
		 * The data will be deallocated by a subsequent call to {@link Deallocator}.
		 *
		 * Clients must provide a custom deallocator function so they can pass in memory managed by something like NumPy.
		 *
		 * @param data_type the {@link DataType} of the values held by the Tensor
		 * @param dims the dimension of the Tensor
		 * @param deallocator deallocatinf function
		 * @see Deallocator
		 */
		[CCode (cname = "TF_NewTensor")]
		public Tensor.with_deallocator (DataType data_type, int64[] dims, uint8[] data, Deallocator deallocator);
		
		/**
		 * Deallocator to be passed to {@link Tensor.with_deallocator}.
		 */
		[CCode (has_target = false)]
		public delegate void Deallocator (uint8[] data);

		/**
		 * Allocate and return a new tensor.
		 *
		 * This function is an alternative of {@link Tensor.with_deallocator} and should be used when memory is allocated to pass the Tensor to the C API.
		 * The allocated memory satisfies TensorFlow's memory alignment preferences and should be preferred over calling ``malloc`` and ``free``.
		 *
		 * @param data_type the {@link DataType} of the values held by the Tensor
		 * @param dims the dimension of the Tensor
		 * @param len the size in bytes of the data held by the Tensor
		 * @see Tensor.with_deallocator
		 */
		[CCode (cname = "TF_AllocateTensor")]
		public Tensor (DataType data_type, long[] dims, size_t len);
		
		/**
		 * Return the {@link DataType} of a Tensor element
		 */
		[CCode (cname = "TF_TensorType")]
		public DataType type ();
		
		/**
		 * Return the number of dimensions that the Tensor has
		 */
		[CCode (cname = "TF_NumDims")]
		public int num_dim ();
		
		/**
		 * Return the length of the tensor in the ``dim_index`` dimension.
		 */
		[CCode (cname = "TF_Dim")]
		public int64 dim (int dim_index);
		
		/**
		 * Return the size of the undelying data in bytes.
		 */
		[CCode (cname = "TF_TensorByteSize")]
		public size_t byte_size ();
		
		/**
		 * Return a pointer to the underlying data.
		 */
		[CCode (cname = "TF_TensorData")]
		public uint8[] data ();
	}
	
	/**
	 * Encode a string in the format required by {@link DataType.STRING} tensors.
	 *
	 * It does not write to memory more than ''dst_len'' bytes beyond ''dst''.
	 *
	 * @param src string to encode
	 * @param src_len length of the string to encode, use src.length
	 * @param dst encoded string in the STRING tensor format
	 * @param dst_len maximum encoded length, it should be at least {@link string_encoded_size} (src_len)
	 * @param status the status code and message of the operation
	 * @return on success: the size in bytes of the encoded string <<BR>>
	 * otherwise: an error into ''status''
	 * @see string_decode
	 * @see string_encoded_size
	 */
	[CCode (cname = "TF_StringEncode")]
	public size_t string_encode (string src, size_t src_len, [CCode (ctype = "char*")] string dst, size_t dst_len, Status status);

	/**
	 * Decode a string encoded using {@link string_encode}
	 *
	 * On success sets dst to the start of the decoded string and dst_len to its length.
	 *
	 * @param src the encoded string
	 * @param src_len length of the encoded string, use src.length
	 * @param dst the decoded string
	 * @param dst_len the length of the decoded string dst
	 * @param status the status code and message of the operation
	 * @return on success: the number of bytes starting at src consumed while decoding <<BR>>
	 * otherwhise: dst and dst_len are undefined and an error is set in status
	 * @see string_encode
	 * @see string_encoded_size
	 */
	[CCode (cname = "TF_StringDecode")]
	public size_t string_decode (string src, size_t src_len, out unowned string dst, out size_t dst_len, Status status);

	/**
	 * Compute the size in bytes to encode a string ''len'' bytes long
	 *
	 * @param len the length of the string to be encoded
	 * @return the size in bytes required to encode a string ''len'' bytes long
	 * @see string_encode
	 */
	[CCode (cname = "TF_StringEncodedSize")]
	public size_t string_encoded_size (size_t len);

	
	/**
	 * Holds options that can be passed during session creation.
	 */
	[CCode (cname = "TF_SessionOptions", free_function = "TF_DeleteSessionOptions")]
	[Compact]
	public class SessionOptions {
		
		/**
		 * Return a new options object.
		 */
		[CCode (cname = "TF_NewSessionOptions")]
		public SessionOptions ();
		
		/**
		 * Set the target in ``SessionOptions.options``.
		 *
		 * ``config`` should be a serialized ``ConfigProto proto``.
		 *
		 * If ``config`` was not parsed successfully as a ``ConfigProto``, record the error information in ``status``.
		 *
		 * @param target string that can be empty. a single entry or a comma separated list of entries.<<BR>>
		 * Each entry is in one of the following formats:
		 * {{{"local"
		 * ip:port
		 * host:port}}}
		 */
		[CCode (cname = "TF_SetTarget")]
		public void set_target (owned string target);
		
		/**
		 * Set the config in ``SessionOptions.options``.
		 *
		 * ``config`` should be a serialized ``ConfigProto proto``.
		 * If ``config`` was not parsed successfully as a ``ConfigProto``, record the error information in ``status``.
		 * 
		 * @param proto the pointer containing the ``config`` to be set
		 */
		[CCode (cname = "TF_SetConfig")]
		public void set_config (uint8[] proto, Status status);
	}

	/**
	 * Represents a specific input of an operation
	 * 
	 * The int field is the index of the input within the Operation
	 */
	[CCode (cname = "TF_Input")]
	public struct Input {
		Operation oper;
		int index;
	}

	/**
	 * Represents a specific output of an operation
	 *
	 * The int field is the index of the output within the Operation
	 */
	[CCode (cname = "TF_Output")]
	public struct Output {
		Operation oper;
		int index;
	}

	/**
	 * Represents a computation graph.
	 *
	 * Graphs may be shared between sessions and are thread-safe.
	 */
	[CCode (cname = "TF_Graph", free_function = "TF_DeleteGraph")]
	[Compact]
	public class Graph {
		[CCode (cname = "TF_NewGraph")]
		public Graph ();

		/**
		 * Sets the shape of the Tensor referenced by ``output`` to the shape described by ``dims``.
		 *
		 * This does not overwrite the existing shape associated with ``output``, but merges the input shape with the existing shape.
		 * For example, setting a shape of ``[-1, 2]`` with an existing shape ``[2, -1]`` would set a final shape of ``[2, 2]`` based on shape merging semantics.
		 *
		 * @param output where the tensor is referenced in this graph
		 * @param dims the dimensions array<<BR>>
		 * It can be null if the number of dimensions is unknown.
		 * If a dimension is unknown, the corresponding entry in the ``dims`` array must be ``-1``.
		 * @param status holds an error if:<<BR>>
		 * * ``output`` is not in this graph
		 * * An invalid shape is being set (e.g., the shape set is incompatible with the existing shape
		 */
		[CCode (cname = "TF_GraphSetTensorShape")]
		public void set_tensor_shape (Output output, int64[]? dims, Status status);
		
		/**
		 * Returns the number of dimensions of the Tensor referenced by ``output`` in this graph.
		 *
		 * @param output where the tensor is referenced in this graph
		 * @param status holds an error if: ``output`` is not in this graph
		 * @return the number of dimensions or ``-1`` if the shape is unknown
		 */
		[CCode (cname = "TF_GraphGetTensorNumDims")]
		public int get_tensor_num_dims (Output output, Status status);
		
		/**
		 * Returns the sape of the Tensor referenced by ``output`` in this graph.
		 *
		 * If the number of dimensions in the shape is unknown or the shape is a scalar, ``dims`` will remain untouched.
		 * Otherwise, each element of ``dims`` will be set corresponding to the size of the dimension.
		 * An unknown dimensions is represented by ``-1``
		 *
		 * @param dims the dimensions array
		 * @param status holds an error if: ``output`` is not in this graph
		 */
		[CCode (cname = "TF_GraphGetTensorShape")]
		public void get_tensor_shape (Output output, int64[] dims, Status status);

		
		/**
		 * Operation will only be added to this graph when ``finish_operation`` is called, assuming it does not returns an error.
		 */
		[CCode (cname = "TF_NewOperation")]
		public OperationDescription new_operation (string op_type, string oper_name);

		/**
		 * Returns the operation in the graph with ``oper_name`` or ``null`` if no operation found.
		 */
		[CCode (cname = "TF_GraphOperationByName")]
		public Operation? graph_operation_by_name (string oper_name);
		
		/**
		 * Iterate through the operations of a graph.
		 *
		 * To use:
		 * {{{
		 * size_t pos;
		 * var oper = new Operation;
		 * while ((oper = oper.next_operation (out pos)) != null) {
		 *     // do something with oper
		 * }
		 * }}}
		 * @param pos the position where the operation is
		 */
		[CCode (cname = "TF_GraphNextOperation")]
		public Operation next_operation (out size_t pos);
		
		/**
		 * Write out a serialized representation of this graph (as a ``graph_def`` protocol message) to ``output_graph_def`` (allocated by {@link Buffer.Buffer}).
		 */
		[CCode (cname = "TF_GraphToGraphDef")]
		public void to_graph_def (Buffer output_graph_def, Status status);
		
		/**
		 * Import the graph serialized in ``graph_def``.
		 */
		[CCode (cname = "TF_GraphImportGraphDef")]
		public void import_graph_def(Buffer graph_def, ImportGraphDefOptions options, Status status);
		
	}

	/**
	 * Operation being built. The underlying graph must outlive this.
	 */
	[CCode (cname = "TF_OperationDescription", destroy_function = "")]
	[Compact]
	public class OperationDescription {

		/**
		 * Specify the defice for ``this``.
		 *
		 * @param device the device for the current OperationDescription. Defaults to empty, meaning unconstrained.
		 */
		[CCode (cname = "TF_SetDevice")]
		public void set_device (string device = "");

		/**
		 * For inputs that take a single tensor.
		 */
		[CCode (cname = "TF_AddInput")]
		public void add_input (Output input);
		[CCode (cname = "TF_AddInputList")]

		/**
		 * For inputs that take a list of tensors.
		 *
		 * @param inputs must point to ``Output[num_inputs]``
		 */
		public void add_input_list (Output[] inputs);

		/**
		 * Call once per control ``input`` to ``this``.
		 */
		[CCode (cname = "TF_AddControlInput")]
		public void add_control_input (Operation input);

		/**
		 * Request that ``this`` be co-located on the device where ``op`` is placed.
		 *
		 * Use of this is discouraged since the implementation of device placement is subject to change.
		 * Primarily intended for internal libraries.
		 */
		[CCode (cname = "TF_ColocateWith")]
		public void colocate_with (Operation op);
		[CCode (cname = "TF_SetAttrString")]
		public void set_attr_string (string attr_name, uint8[] value);
		[CCode (cname = "TF_SetAttrStringList")]
		public void set_attr_string_list (string attr_name, [CCode (array_length = false)] uint8[] values, int[] lengths);
		[CCode (cname = "TF_SetAttrInt")]
		public void set_attr_int (string attr_name, int64 value);
		[CCode (cname = "TF_SetAttrIntList")]
		public void set_attr_int_list (string attr_name, int64[] values);
		[CCode (cname = "TF_SetAttrFloat")]
		public void set_attr_float (string attr_name, float value);
		[CCode (cname = "TF_SetAttrFloatList")]
		public void set_attr_float_list (string attr_name, float[] values);
		[CCode (cname = "TF_SetAttrBool")]
		public void set_attr_bool (string attr_name, uchar value);
		[CCode (cname = "TF_SetAttrBoolList")]
		public void set_attr_bool_list (string attr_name, uchar[] values);
		[CCode (cname = "TF_SetAttrType")]
		public void set_attr_type (string attr_name, DataType value);
		[CCode (cname = "TF_SetAttrTypeList")]
		public void set_attr_type_list (string attr_name, DataType[] values);
		[CCode (cname = "TF_SetAttrShape")]
		public void set_attr_shape (string attr_name, int64[] dims);
		[CCode (cname = "TF_SetAttrShapeList")]
		public void set_attr_shape_list (string attr_name, [CCode (array_length = false)] int64[] dims, int[] num_dims);
		[CCode (cname = "SetAttrTensorShapeProto")]
		public void set_attr_tensor_shape_proto (string attr_name, void[] proto, Status status);
		[CCode (cname = "SetAttrTensorShapeProtoList")]
		public void set_attr_tensor_shape_proto_list (string attr_name, [CCode (array_length = false)] uint8[] protos, int[] proto_lens, Status status);
		[CCode (cname = "TF_SetAttrTensor")]
		public void set_attr_tensor (string attr_name, Tensor value, Status status);
		[CCode (cname = "TF_SetAttrTensorList")]
		public void set_attr_tensor_list (string attr_name, Tensor[] values, Status status);
		[CCode (cname = "TF_SetAttrValueProto")]
		public void set_attr_value_proto (string attr_name, void[] proto, Status status);
		[CCode (cname = "TF_FinishOperation")]
		public void finish_operation (Status status);
	}

	/**
	 * Operations are immutable once created, so these are all query methods.
	 */
	[CCode (cname = "TF_Operation", destroy_function = "")]
	public class Operation {
		[CCode (cname = "TF_OperationName")]
		public unowned string name ();
		[CCode (cname = "TF_OperationOpType")]
		public unowned string op_type (Port oper_in);
		[CCode (cname = "TF_OperationDevice")]
		public unowned string device ();
		[CCode (cname = "TF_OperationNumOutputs")]
		public int num_outputs ();
		[CCode (cname = "TF_OperationOutputType")]
		public DataType output_type (Port oper_out);
		[CCode (cname = "TF_OperationOutputListLength")]
		public int output_list_length (string arg_name, Status status);
		[CCode (cname = "TF_OperationNumInputs")]
		public int num_inputs ();
		[CCode (cname = "TF_OperationInputType")]
		public DataType input_type (Port oper_in);
		[CCode (cname = "TF_OperationInputListLength")]
		public int input_list_length (string arg_name, Status status);
		[CCode (cname = "TF_OperationInput")]
		extern Port input(Port oper_in);
		[CCode (cname = "TF_OperationOutputNumConsumers")]
		extern int output_num_consumers (Port oper_out);
		[CCode (cname = "TF_OperationOutputConsumers")]
		public int operation_output_consumers (Port oper_out, Port[] consumers);
		[CCode (cname = "TF_OperationNumControlInputs")]
		public int num_control_inputs ();
		[CCode (cname = "TF_OperationGetControlInputs")]
		public int get_control_inputs (Operation[] control_inputs);
		[CCode (cname = "TF_OperationNumControlOutputs")]
		public int num_control_outputs ();
		[CCode (cname = "TF_OperationGetControlOutputs")]
		public int get_control_outputs (Operation[] control_outputs);
		[CCode (cname = "TF_OperationGetAttrMetadata")]
		public AttrMetadata get_attr_metadata (string attr_name, Status status);
		[CCode (cname = "TF_OperationGetAttrString")]
		public void get_attr_string (string attr_name, void[] value, Status status);
		[CCode (cname = "TF_OperationGetAttrStringList")]
		public void get_attr_string_list (string attr_name, uint8[,] values, uint8[] storage, Status status);
		[CCode (cname = "TF_OperationGetAttrInt")]
		public void get_attr_int (string attr_name, out int64 value, Status status);
		[CCode (cname = "TF_OperationGetAttrIntList")]
		public void get_attr_int_list (string attr_name, out int64 values, int max_values, Status status);
		[CCode (cname = "TF_OperationGetAttrFloat")]
		public void get_attr_float (string attr_name, out float value, Status status);
		[CCode (cname = "TF_OperationGetAttrFloatList")]
		public void get_attr_float_list (string attr_name, out float values, int max_values, Status status);
		[CCode (cname = "TF_OperationGetAttrBool")]
		public void get_attr_bool (string attr_name, out uchar value, Status status);
		[CCode (cname = "TF_OperationGetAttrBoolList")]
		public void get_attr_bool_list (string attr_name, out uchar values, int max_values, Status status);
		[CCode (cname = "TF_OperationGetAttrType")]
		public void get_attr_type (string attr_name, out DataType value, Status status);
		[CCode (cname = "TF_OperationGetAttrTypeList")]
		public void get_attr_type_list (string attr_name, out DataType values, int max_values, Status status);
		[CCode (cname = "TF_OperationGetAttrShape")]
		public void get_attr_shape (string attr_name, int64* value, int num_dims, Status status);
		[CCode (cname = "TF_OperationGetAttrShapeList")]
		public void get_attr_shape_list (string attr_name, int64[,] dims, int64[] storage, Status status);
		[CCode (cname = "TF_OperationGetAttrTensorShapeProto")]
		public void get_attr_tensor_shape_proto (string attr_name, Buffer value, Status status);
		[CCode (cname = "TF_OperationGetAttrTensorShapeProtoList")]
		public void get_attr_tensor_shape_proto_list (string attr_name, Buffer[] values, Status status);
		[CCode (cname = "TF_OperationGetAttrTensor")]
		public void get_attr_tensor (string attr_name, out Tensor value, Status status);
		[CCode (cname = "TF_OperationGetAttrTensorList")]
		public void get_attr_tensor_list (string attr_name, out Tensor[] values, Status status);
		[CCode (cname = "TF_OperationGetAttrValueProto")]
		public void get_attr_value_proto (string attr_name, Buffer output_attr_value, Status status);
		[CCode (cname = "TF_OperationToNodeDef")]
		extern void to_node_def (Buffer output_node_def, Status status);
	}

	[CCode (cname = "TF_Port", destroy_function = "", has_type_id = false)]
	[SimpleType]
	public struct Port {
		Operation oper;
		int index;
	}

	[CCode (cname = "TF_AttrType", cprefix = "TF_ATTR_", has_type_id = false)]
	public enum AttrType {
		STRING = 0,
		INT = 1,
		FLOAT = 2,
		BOOL = 3,
		TYPE = 4,
		SHAPE = 5,
		TENSOR = 6,
		PLACEHOLDER = 7,
		FUNC = 8,
	}

	[CCode (cname = "TF_AttrMetadata", destroy_function = "", has_type_id = false)]
	public struct AttrMetadata {
		uchar is_list;
		int64 list_size;
		AttrType type;
		int64 total_size;
	}

	[CCode (cname = "TF_ImportGraphDefOptions", free_function = "TF_DeleteImportGraphDefOptions")]
	[Compact]
	public class ImportGraphDefOptions {
		[CCode (cname = "TF_NewImportGraphDefOptions")]
		public ImportGraphDefOptions ();
		[CCode (cname = "TF_ImportGraphDefOptionsSetPrefix")]
		public void set_prefix (string prefix);
	}

	[CCode (cname = "TF_SessionWithGraph", free_function = "")]
	[Compact]
	public class SessionWithGraph {
		[CCode (cname = "TF_NewSessionWithGraph")]
		public SessionWithGraph (Graph graph, SessionOptions opts, Status status);
		[CCode (cname = "TF_CloseSessionWithGraph")]
		public void close (Status status);
		[CCode (cname = "TF_DeleteSessionWithGraph")]
		public void delete (Status status);
		[CCode (cname = "TF_SessionRun")]
		public Status run (
			// RunOptions
			Buffer run_options,
			// Input tensors
			[CCode (array_length = false)] Port inputs, Tensor[] input_values,
			// Output tensors
			[CCode (array_length = false)] Port outputs, Tensor[] output_values,
			// Target operations
			Operation[] target_opers,
			// RunMetadata
			Buffer run_metadata);
		[CCode (cname = "TF_SessionPRunSetup")]
		public Status session_p_run_setup (
			// Input names
			Port[] inputs,
			// Output names
			Port[] outputs,
			// Target operations
			Operation[] target_opers,
			// Output handle
			string* handle);
		[CCode (cname = "TF_SessionPRun")]
		public void session_p_run (string handle,
								   // Input tensors
								   [CCode (array_length = false)] Port inputs, Tensor[] input_values,
								   // Output tensors
								   [CCode (array_length = false)] Port outputs, Tensor[] output_values,
								   // Target operations
								   Operation[] target_opers);
	}

	[CCode (cname = "TF_Session", free_function = "TF_DeleteSession")]
	[Compact]
	public class Session {
		[CCode (cname = "TF_NewSession")]
		public Session (SessionOptions opts, Status status);
		[CCode (cname = "TF_LoadSessionFromSavedModel")]
		public Session load_session_from_saved_model (SessionOptions session_options,
													  Buffer run_options,
													  string export_dir,
													  string[] tags,
													  Graph graph,
													  Buffer meta_graph_def,
													  Status status);
		[CCode (cname = "TF_CloseSession")]
		public Status close (Status status);
		/* [CCode (cname = "TF_DeleteSession")] */
		/* public Status delete (Status status); */
		[CCode (cname = "TF_Reset")]
		public void reset (SessionOptions opt, string[] containers, Status status);
		[CCode (cname = "TF_ExtendGraph")]
		public void extend_graph (void[] proto, Status status);
		[CCode (cname = "TF_Run")]
		public Status run (
			// RunOptions
			Buffer run_options,
			// Input tensors
			[CCode (array_length = false)] string input_names, Tensor[] inputs,
			// Output tensors
			[CCode (array_length = false)] string output_names, Tensor[] outputs,
			// Target operations
			[CCode (array_length = false)] string target_oper_names[],
			// RunMetadata
			Buffer run_metadata);
		[CCode (cname = "TF_PRunSetup")]
		extern Status p_run_setup (
			// Input names
			string[] input_names,
			// Output names
			string[] output_names,
			// Target operations
			string[] target_oper_names,
			// Output handle
			out string handle);
		[CCode (cname = "TF_PRun")]
		extern Status p_run (string handle,
							 // Input tensors
							 [CCode (array_length = false)] string input_names, Tensor[] inputs,
							 // Output tensors
							 [CCode (array_length = false)] string output_names, Tensor[] outputs,
							 // Target operations
							 string[] target_oper_names);
	}

	[CCode (cname = "TF_Library", free_function = "")]
	[Compact]
	public class Library {
		[CCode (cname = "TF_LoadLibrary")]
		public Library.load (string library_filename, Status status);
		[CCode (cname = "TF_GetOpList")]
		public Buffer get_op_list (Library lib_handle);
		[CCode (cname = "TF_DeleteLibraryHandle")]
		public void delete ();
		[CCode (cname = "TF_GetAllOpList")]
		public Buffer get_all_op_list ();
	}
}
