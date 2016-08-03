from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import tensor_array_ops #

def dynamic_rnn_decoder(decoder_inputs,cell,initial_state = None,dtype=dtypes.float32,sequence_length=None,
						loop_function=None,parallel_iterations=None,
						swap_memory=False,time_major=False,scope=None):
	"""RNN decoder for seq2seq model.
	This function is functionally identical to the function `rnn_decoder` above, but performs fully dynamic unrolling of `inputs`.
	Unlike `rnn_decoder`, the input `inputs` is not a Python list of `Tensors`. Instead it is a single `Tensor` where the maximum time is either the first or second dimension (see the parameter `time_major`). The corresponding output is a single `Tensor` having the same number of time steps and batch size.

	The parameter `sequence_length` is required and dynamic calculation is automatically performed.

	Args:
		decoder_inputs: the RNN decoder inputs.
		If time_major == False (default), this must be a tensor of shape:
		`[batch_size,max_time,cell_input_size]`
		If time_major == True, this must be a tensor of shape:
		`[max_time,batch_size,cell.input_size]`
	"""
	if not isinstance(cell,rnn_cell.RNNCell):
		raise TypeError("cell must be an instance of RNNCell")
	if not time_major:
		inputs = array_ops.transpose(decoder_inputs,[1,0,2])
	parallel_iterations = parallel_iterations or 32
	if sequence_length is not None:
		sequence_length = math_ops.to_int32(sequence_length)
		sequence_length = array_ops.identity(sequence_length,name="sequence_length")
	with variable_scope.variable_scope(scope or "dynamic_rnn_decoder") as varscope:
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		outputs,state = _dynamic_rnn_decoder_loop(inputs,initial_state,cell,sequence_length,loop_function,parallel_iterations,swap_memory,dtype)

		if not time_major:
			outputs = array_ops.transpose(outputs,[1,0,2])
		return outputs,state

def _dynamic_rnn_decoder_loop(inputs,initial_state,cell,sequence_length,loop_function,parallel_iterations,swap_memory,dtype):
	"""Internal implementation of Dynamic RNN decoder.
	"""
	assert isinstance(parallel_iterations,int),"parallel_iterations must be int"
	input_shape = array_ops.shape(inputs)
	time_steps, batch_size, _ = array_ops.unpack(input_shape,3)
	inputs_got_shape = inputs.get_shape().with_rank(3)
	const_time_steps, const_batch_size, const_depth = inputs_got_shape.as_list()

	if const_depth is None:
		raise ValueError("Input size (depth of inputs) must be accessible via shape inference but saw value None")

	zeros_ouput = array_ops.zeros(array_ops.pack([batch_size, cell.output_size]), inputs.dtype)
	
	with ops.op_scope([],"dynamic_rnn_decoder_loop") as scope:
		base_name = scope

	output_ta = tensor_array_ops.TensorArray(dtype=inputs.dtype, size=time_steps, clear_after_read = False, tensor_array_name=base_name+"output")
	input_ta = tensor_array_ops.TensorArray(dtype=inputs.dtype, size=time_steps, tensor_array_name=base_name+"input")

	input_ta = input_ta.unpack(inputs)
	time = array_ops.constant(1, dtype=dtypes.int32, name="time")
	with variable_scope.variable_scope("decoding") as scope:
		GO = input_ta.read(0)
		GO.set_shape([const_batch_size, const_depth])
		if initial_state is None:
			initial_state = cell.zero_state(batch_size, dtype)
		output_0, state_0 = cell(GO, initial_state)
		output_ta_0 = output_ta.write(0,output_0)


	def _time_step(time, state, output_ta_t):
		"""Take a time step of the dynamic RNN decoder.
		"""
		input_t = input_ta.read(time)
		input_t.set_shape([const_batch_size, const_depth])
		if loop_function is not None:
			output_t_minus_1 = output_ta_t.read(time-1)
			output_t_minus_1.set_shape([const_batch_size,cell.output_size])
			input_t = loop_function(input_t,output_t_minus_1,time)
		
		with variable_scope.variable_scope(scope,reuse=True):
			_output, _state = cell(input_t, state)

		if sequence_length is not None:
			copy_cond = (time>=sequence_length)
			new_output = math_ops.select(copy_cond, zeros_ouput, _output)
			new_state = math_ops.select(copy_cond, state, _state)
		else:
			new_output, new_state = _output, _state
		output_ta_t = output_ta_t.write(time, new_output)
		
		return (time+1, new_state, output_ta_t)

	(_, final_state, final_output_ta) = control_flow_ops.while_loop(
		cond = lambda time, _1, _2: time < time_steps,
		body = _time_step,
		loop_vars = (time, state_0, output_ta_0),
		parallel_iterations = parallel_iterations,
		swap_memory = swap_memory)

	final_outputs = final_output_ta.pack()
	final_outputs.set_shape([const_time_steps, const_batch_size, cell.output_size])

	return (final_outputs,final_state)