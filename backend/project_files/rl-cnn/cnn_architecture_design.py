import tensorflow as tf
from tensorflow.keras import layers, models

conv_functions  = ["Conv2D"]
conv_num_filters  = ["32", "64"]
conv_filter_sizes  = ["3", "5", "7"]
conv_activation_functions = ['relu']
conv_paddings  = ["same"]

pool_functions  = ["MaxPooling2D"]
pool_filter_sizes  = ["2"]
pool_strides  = ["2"]

dense_num_neurons  = ["128", "256", "512", "1024"]
dense_activation_functions  = ["relu"]

learning_rates  = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
optimizer_names  = ["adam"]

dropout_rates=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

conv_layers = []
for conv_function in conv_functions:
	for conv_num_filter in conv_num_filters:
		for conv_filter_size in conv_filter_sizes:
			for conv_activation_function in conv_activation_functions:
				for conv_padding in conv_paddings:
					conv_layer = "layers."+conv_function+"("+conv_num_filter+", ("+conv_filter_size+", "+conv_filter_size+"), activation='"+conv_activation_function+"', input_shape=(9, 9, 1), padding='"+conv_padding+"')"
					conv_layers.append(conv_layer)

pool_layers = []
for pool_function in pool_functions:
	for pool_filter_size in pool_filter_sizes:
		for pool_stride in pool_strides:
			pool_layer = "layers."+pool_function+"(("+pool_filter_size+", "+pool_filter_size+"), strides="+pool_stride+")"
			pool_layers.append(pool_layer)

dense_layers = []
for dense_num_neuron in dense_num_neurons:
	for dense_activation_function in dense_activation_functions:
					dense_layer = "layers.Dense("+dense_num_neuron+", activation='"+dense_activation_function+"')"
					dense_layers.append(dense_layer)

optimizers = []
for optimizer_name in optimizer_names:
	for learning_rate in learning_rates:
		if optimizer_name == "adam":
			optimizer = tf.optimizers.Adam(learning_rate)
		optimizers.append(optimizer)

dropout_layers = []
for dropout_rate in dropout_rates:
	dropout_layer = "layers.Dropout("+dropout_rate+")"
	dropout_layers.append(dropout_layer)

# Action
def generate_selective_actions(num_conv_layer):
	include_conv = True
	include_pool = False
	include_dense = True
	include_optimzier = True
	include_dropout = True

	actions = []
	dense_layer_position = num_conv_layer * 3
	dropout_after_dense_layer_position = dense_layer_position + 1
	optimizer_position = dropout_after_dense_layer_position + 1
	if include_conv:
		i = 0
		for _ in range(num_conv_layer):
			for j in range(len(conv_layers)):
				actions.append([i, j])
			i += 3
	if include_pool:
		i = 1
		for _ in range(num_conv_layer):
			for j in range(len(pool_layers)):
				actions.append([i, j])
			i += 3
	if include_dropout:
		i = 2
		for _ in range(num_conv_layer):
			for j in range(len(dropout_layers)):
				actions.append([i, j])
			i += 3
		for j in range(len(dropout_layers)):
			actions.append([dropout_after_dense_layer_position, j])
	if include_dense:
		for j in range(len(dense_layers)):
			actions.append([dense_layer_position, j])
	if include_optimzier:
		for j in range(len(optimizers)):
			actions.append([optimizer_position, j])
	return actions

def generate_cnn_model(state):
	num_classes = 8

	# Build architecture
	model = models.Sequential()

	conv_layer_1 = conv_layers[state[0]]
	pool_layer_1 = pool_layers[state[1]]
	dropout_layer_1 = dropout_layers[state[2]]
	conv_layer_2 = conv_layers[state[3]]
	pool_layer_2 = pool_layers[state[4]]
	dropout_layer_2 = dropout_layers[state[5]]
	dense_before_dropout_layer = dense_layers[state[6]]
	dropout_before_output_layer = dropout_layers[state[7]]

	model.add(eval(conv_layer_1))
	model.add(eval(pool_layer_1))
	model.add(eval(dropout_layer_1))
	model.add(eval(conv_layer_2))
	model.add(eval(pool_layer_2))
	model.add(eval(dropout_layer_2))
	model.add(layers.Flatten())
	model.add(eval(dense_before_dropout_layer))
	model.add(eval(dropout_before_output_layer))	
	model.add(layers.Dense(num_classes))

	model.compile(optimizers[state[8]],
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])
	return model
