import tensorflow as tf

def def_cell(args, rnn_size):
        if args.cell_model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.cell_model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif args.cell_model == 'lstm':
            pass
        else:
            raise Exception("model type not supported: {}".format(args.cell_model))
 
        cell_ = cell_fn(rnn_size, reuse=tf.get_variable_scope().reuse)
        if args.keep_prob < 1.:
            cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=args.keep_prob)
        return cell_

def extract_feature(x, args, reuse=False):
    with tf.variable_scope("Word_Level_CNN", reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()
            #print(scope.name)
        else:
            assert tf.get_variable_scope().reuse == False

        with tf.variable_scope("Embedding", reuse=reuse):
            splitted_word_ids  = tf.split(x, args.max_time_step, axis=1)
            embedding_weight = tf.get_variable(name='embedding_weight', shape=[args.vocab_size, args.embedding_size])
            t_embedded = []
            
            for t in range(args.max_time_step):
                if t is not 0:
                    tf.get_variable_scope().reuse_variables()

                embedded = tf.nn.embedding_lookup(embedding_weight, x[:,t,:])
                t_embedded.append(embedded)
            cnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(t_embedded), perm=(1,0,2,3)), (-1, args.max_time_step, args.embedding_size,1))
           
        kernels = [2,3,4,5,6]
        filter_nums = [32,64,128,128,224]
        with tf.variable_scope("CNN", reuse=reuse):
            convded = []
            for kernel, filter_num in zip(kernels, filter_nums):
                conv_ = tf.layers.conv2d(cnn_inputs, filter_num, kernel_size=[kernel, args.embedding_size], strides=[1, 1], activation=tf.nn.relu, padding='valid', name="conv_{}".format(kernel), reuse=reuse)
                pool_ = tf.layers.max_pooling2d(conv_, pool_size=[args.max_time_step-kernel+1, 1], padding='valid', strides=[1, 1])
                convded.append(tf.reshape(pool_, (-1, filter_num)))
            convded = tf.concat([cnn_output for cnn_output in convded], axis=-1)
    return  tf.contrib.layers.flatten(convded)

def converter(x, x_idx, args, name, reuse=False, extract_reuse=False):
    ##using word level cnn's feature
    ##input shapes are (None, max_timestep, 1)
    ##output shape are (None, max_timestep, 1) and return index which is highest probablistic.
    extracted_feature = extract_feature(x_idx, args, extract_reuse)
    #extracted_feature = tf.reshape(tf.convert_to_tensor([extracted_feature for _ in range(4)]),(-1, 4*args.gen_rnn_size))
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope(name+'Embedding') as scope:
            rnn_inputs = []
            embedding_weight = tf.get_variable(shape=[args.vocab_size, args.rnn_embedding_size], name='embedding_weight')
                
            for t in range(args.max_time_step):
                embedded = tf.nn.embedding_lookup(embedding_weight, x_idx[:,t,:])
                rnn_inputs.append(embedded)
            rnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(rnn_inputs), (0,1,3,2)), (-1, args.max_time_step, args.embedding_size))
            
        with tf.variable_scope(name+'RNN') as scope:
            #cell_ = tf.contrib.rnn.MultiRNNCell([def_cell(args, args.gen_rnn_size) for _ in range(1)], state_is_tuple = True)     
            cell_ = def_cell(args, args.gen_rnn_size)
            rnn_outs, _ = tf.nn.dynamic_rnn(cell_, rnn_inputs, initial_state=extracted_feature, dtype=tf.float32)

        with tf.variable_scope(name+"Dense") as scope:
            logits = []
            outputs = []
            indexs = []
            for t in range(args.max_time_step):
                if t != 0:
                    tf.get_variable_scope().reuse_variables()
                logit = tf.layers.dense(rnn_outs[:,t,:], args.vocab_size, activation=tf.nn.sigmoid, name="rnn_dense")
                out = tf.nn.softmax(logit)
                logits.append(logit)
                outputs.append(out)
                indexs.append(tf.to_int32(tf.expand_dims(tf.argmax(out, axis=-1), axis=-1)))
            logits = tf.transpose(tf.convert_to_tensor(logits), (1,0,2))
            outputs = tf.transpose(tf.convert_to_tensor(outputs), (1,0,2))
            indexs = tf.transpose(tf.convert_to_tensor(indexs), (1,0,2))
        return logits, outputs, indexs

def discriminator(x, args, name, reuse=False): 
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope(name+"RNN") as scope:
            cell_ = def_cell(args, args.dis_rnn_size)
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell_, x, initial_state=cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32), dtype=tf.float32) 

        with tf.variable_scope(name+"Dense") as scope:
            if args.merged_all:
                outputs = []
                for t in range(args.max_time_step):
                    if t != 0:
                        tf.get_variable_scope().reuse_variables()

                    outputs.append(tf.layers.dense(rnn_outputs[:,t,:], 1, activation=tf.nn.sigmoid, name="rnn_out_dense"))
                logits = tf.reduce_sum(tf.transpose(tf.convert_to_tensor(outputs),(1,0,2)), axis=1)
            else:
                logits = tf.layers.dense(rnn_outputs[-1], 1, activation=tf.nn.sigmoid)

        return logits
