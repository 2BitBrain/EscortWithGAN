import tensorflow as tf

def def_cell(args, rnn_size, reuse=False):
        if args.cell_model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.cell_model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif args.cell_model == 'lstm':
            pass
        else:
            raise Exception("model type not supported: {}".format(args.cell_model))
 
        cell_ = cell_fn(rnn_size, reuse=reuse)
        #if args.keep_prob < 1.:
        #    cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=args.keep_prob)
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

def decoder(x, args, cell, state, activation=tf.nn.sigmoid):
    logits = []
    probablistic = []
    indexs = []
    for t in range(args.max_time_step):
        if t != 0:
            tf.get_variable_scope().reuse_variables()

        out, state = cell(x, state)
        logit = tf.layers.dense(out, args.vocab_size, activation=activation, name="rnn_out_dense")
        logits.append(logit)
        probablistic.append(tf.nn.softmax(logit))
        indexs.append(tf.argmax(tf.nn.softmax(logit), axis=-1))
        x =tf.nn.softmax(logit) 
    logits = tf.transpose(tf.convert_to_tensor(logits), (1,0,2))
    prob = tf.transpose(tf.convert_to_tensor(probablistic), (1,0,2))
    indexs = tf.expand_dims(tf.transpose(tf.convert_to_tensor(indexs), (1,0)), axis=-1)
    return logits, prob, indexs
                          












class Generator():
    def __init__(x, p_e_x, p_d_x, go, args, name, reuse=False, extract_reuse=False):
        extracted_feature = extract_feature(x, args, extract_reuse)*0.01
        with tf.variable_scope(name, reuse=reuse) as scope:
            #pre training
            p_rnn_inputs = []
            if args.encoder_embedding:
                embedding_weight = tf.get_variable(shape=[args.vocab_size, args.embedding_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, name='embedding_weight')
                for t in range(args.max_time_step):
                    embedded = tf.nn.embedding_lookup(embedding_weight, p_e_x[:,t,:])
                    p_rnn_inputs.append(embedded)
                rnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(rnn_inputs), (1,0,3,2)), (-1, args.max_time_step, args.embedding_size))
            else:
                for t in range(args.max_time_step):
                    if t!= 0:
                        scope.reuse_variables()        
                    p_rnn_inputs.append(tf.layers.dense(p_e_x[:,t,:], self.args.embedding_size, tf.nn.relu, name="embedding_dense"))
                rnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(rnn_inputs), (1,0,2)))
            
            encoder_cell = def_cell(args, args.gen_rnn_size)
            _, final_state = tf.nn.dynamic_rnn(encoder_cell, rnn_inputs, initial_state=encoder_cell.zero_state(batch_size=args.batch_size, dtype=tf.float32), dtype=tf.float32)
            
            if args.use_extracted_feature:
                decoder_cell = def_cell(args, args.gen_rnn_size + 576)
                state = tf.concat([final_state, extracted_feature], axis=-1)
                noise = tf.random_normal(shape=tf.shape(state), mean=0., stddev=1., dtype=tf.float32)
                state = tf.nn.tanh(state+noise)
            else:
                state = final_state
                decoder_cell = def_cell(args, args.gen_rnn_size)
            
            outputs = []
            out = go
            if args.decoder_embedding:
                d_embedding_weight = tf.get_variable(shape=[args.vocab_size, args.embedding_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, name= "d_embedding_weight")
                for t in range(args.max_time_step):
                    if t != 0:
                        scope.reuse_variables()

                    d_embedded = tf.nn.embedding_lookup(d_embedding_weight, out)
                    rnn_output_, state = decoder_cell(input_, state)
                    out_ = tf.argmax(tf.layers.dense(rnn_output_, args.vocab_size, name="rnn_out_dense"), axis=-1)
                    out = tf.argmax(out_, axis=-1)
                    outputs.append(out_)
            else:
                for t in range(args.max_time_step):
                    if t != 0:
                        scope.reuse_variables()

                    input_ = tf.layers.dense(out, self.args.embedding_size, tf.nn.relu, "decoder_embedding_dense")
                    rnn_output_, state = decoder_cell(input_, state)
                    out = tf.layers.dense(rnn_output_, args.vocab_size, name="rnn_out_dense")
                    outputs.append(out)
            self.p_outputs = tf.transpose(outputs, (1,0,2))

            scope.reuse_variables()
            #training
            rnn_inputs = []
            if args.embedding:
                embedding_weight = tf.get_variable(shape=[args.vocab_size, args.embedding_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, name= "embedding_weight")

                for t in range(args.max_time_step):
                    embedded = tf.nn.embedding_lookup(embedding_weight, x[:,t,:])
                    rnn_inputs.append(embedded)
                rnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(rnn_inputs), (1,0,3,2)),(-1, args.max_time_step, args.embedding_size))
            else:
                for t in range(args.max_time_step):
                    if t != 0:
                        scope().reuse_variables()

                    rnn_inputs.append(tf.layers.dense(x[:,t,:], self.args.embedding_size, tf.nn.relu, name="embedding_dense"))
                rnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(rnn_inputs), (1,0,2)))

            _, final_state = tf.nn.dynamic_rnn(encoder_cell, rnn_inputs, initial_state=encoder_cell.zero_state(batch_size=args.batch_size, dtype=tf.float32), dtype=tf.float32)
        
            if args.use_extracted_feature:

            else:
        
            outputs = []
            out = go
            for t in range(args.max_time_step):
                if t!= 0:
                    scope.reuse_variables()

                input_ = tf.layers.dense(out, self.args.embedding_size, tf.nn.relu, "decoder_embedding_dense")
                rnn_output_, state_ = cell(input_, state)
                output_ = tf.nn.softmax(tf.layers.dense(rnn_output_, args.vocab_size, name="rnn_out_dense"))
                outputs.append(output_)
            self.outputs = tf.transpose(outputs, (1,0,2))

def discriminator(x, args, name, reuse=False): 
    with tf.variable_scope(name, reuse=reuse) as scope:

        print(tf.contrib.framework.get_name_scope())
        with tf.variable_scope(name+"RNN", reuse=reuse) as scope:
            if args.cell_model == 'rnn':
                cell_fn = tf.contrib.rnn.BasicRNNCell
            elif args.cell_model == 'gru':
                cell_fn = tf.contrib.rnn.GRUCell
            elif args.cell_model == 'lstm':
                pass
            else:
                raise Exception("model type not supported: {}".format(args.cell_model))
 
            cell_ = cell_fn(args.dis_rnn_size, reuse=reuse)
            #if args.keep_prob < 1.:
            #    cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=args.keep_prob)
        
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell_, x, initial_state=cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32), scope=name+"d_rnn",dtype=tf.float32) 

        with tf.variable_scope(name+"Dense", reuse=reuse) as scope:
            if args.merged_all:
                outputs = []
                for t in range(args.max_time_step):
                    if t != 0:
                        tf.get_variable_scope().reuse_variables()

                    outputs.append(tf.layers.dense(rnn_outputs[:,t,:], 1, activation=tf.nn.sigmoid, name="rnn_out_dense"))
                logits = tf.reduce_sum(tf.transpose(tf.convert_to_tensor(outputs),(1,0,2)), axis=1)
            else:
                logits = tf.layers.dense(rnn_outputs[-1], 1, activation=tf.nn.sigmoid, reuse=reuse)

        return logits, final_state
