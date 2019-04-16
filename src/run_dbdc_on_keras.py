# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""
A bert-japanese fine-tuning script
using keras-bert

Before running, 
pip install keras-bert sentencepiece
"""

import os
import argparse
import collections
import json
import csv

import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.contrib.tpu.python.tpu import keras_support

import sys
repo_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(repo_path, 'thirdparties'))
from tf_keras_bert import load_trained_model_from_checkpoint
from tf_keras_bert import get_custom_objects

#import tokenization_sentencepiece
import tokenization_sp_mod as tokenization


class DefaultProcessor(object):
    """
    data preprocessor for dbdc corpus
    """
    
    @staticmethod
    def get_rows(file_path):
        rows = []
        with tf.gfile.Open(file_path, 'r') as f:
            for row in csv.reader(f):
                rows.append(row)
        return rows
  
    def get_labels(self):
        return ['O', 'T', 'X']
  
    def get_train_examples(self, data_dir):
        return self._create_examples(
                self.get_rows(os.path.join(data_dir, "train.csv")), 
                'train'
           )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
                self.get_rows(os.path.join(data_dir, "dev.csv")),
                'dev'
            )

    def get_test_examples(self, data_dir):
        return self._create_examples(
                self.get_rows(os.path.join(data_dir, "test.csv")), 
                'test'
            )
  
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                idx_text_a = line.index('text_a')
                idx_text_b = line.index('text_b')
                idx_prob = line.index('prob')
            else:
                guid = "%s-%s" % (set_type, i)
                text_a = tokenization.convert_to_unicode(line[idx_text_a])
                text_b = tokenization.convert_to_unicode(line[idx_text_b])
                prob = json.loads(line[idx_prob])
                prob = [prob.get(label, 0) for label in self.get_labels()]
                examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=prob)
                    )
        return examples


def build_model(args):
    """
    load pretrained bert model and
    append custom layers
    returns keras model
    """
    
    bert_config = {}
    with open(args.bert_config_path, 'r') as f:
        bert_config = json.load(f)
    
    last_hidden_layer_name = 'Encoder-%d-FeedForward-Norm'%(
                bert_config['num_hidden_layers']
            )
    use_layer_name = args.use_layer_name or last_hidden_layer_name
    len_labels = len(args.processor.get_labels())
    
    trained_bert = load_trained_model_from_checkpoint(
            args.bert_config_path, 
            args.init_checkpoint,
            training=True,
        )
    
    last_hidden_output = trained_bert.get_layer(name=use_layer_name).output
    cls_output = L.Lambda(lambda x: x[:, 0])(last_hidden_output)
    cls_output = L.Dropout(args.dropout_rate)(cls_output)
    custom_output = L.Dense(len_labels, activation='softmax')(cls_output)
    model = Model(inputs=trained_bert.input,outputs=custom_output)
   
    optimizer = keras.optimizers.Adam(
            lr=args.learning_rate,
            beta_1=0.9, 
            beta_2=0.999, 
            decay=0.0, 
        )
    model.compile(
            optimizer=optimizer, 
            loss='kullback_leibler_divergence', 
            metrics=['accuracy'],
        )
    model.summary(line_length=156)
    
    return model


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """
    
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                    sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                    Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                   specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """
    Fake example so the num input examples is a multiple of the batch size.
    
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """
    pass


class InputFeatures(object):
    """
    A single set of features of data.
    """
    
    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            label,
            is_real_example=True
        ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.is_real_example = is_real_example


def from_example_to_features(
        ex_index, 
        example, 
        label_list, 
        max_seq_length,
        tokenizer
    ):  
  
    if isinstance(example, PaddingInputExample):
        label_uniform = [1.0/len(label_list)]*len(label_list)
        return InputFeatures(
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label=label_uniform,
                is_real_example=False
            )
        
    use_token_b = example.text_b is not None
    
    # separete input texts into tokens
    tokens_a = tokenizer.tokenize(example.text_a)
    
    tokens_b = []
    if use_token_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    
    truncate_tokens(tokens_a, tokens_b, max_seq_length)
    
    # concatenate tokens
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segment_ids = [0]*(len(tokens_a)+2)
    
    if use_token_b:
        tokens += tokens_b + ['[SEP]']
        segment_ids += [1]*(len(tokens_b)+1)
  
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*len(input_ids)

    # Zero-pad up to the sequence length.
    input_ids += [0]*(max_seq_length - len(input_ids))
    input_mask += [0]*(max_seq_length - len(input_mask))
    segment_ids += [0]*(max_seq_length - len(segment_ids))

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
  
    # label is expected to be a probability distribution
    # so to change example.label when its type is str
    if isinstance(example.label, str):
       label_map = {}
       for (i, label) in enumerate(label_list):
           label_map[label] = i
       label = [0.0]*len(label_list)
       label[label_map[example.label]] = 1.0
    else:
       label = example.label   
  
    # debug output
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]
            ))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (dist = %s)" % (example.label, label))
  
    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label=label,
            is_real_example=True
        )
    return feature


def truncate_tokens(tokens_a, tokens_b, max_length, pop_back_a=True, pop_back_b=False):

    if tokens_b:
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _max = max_length - 3
        targets = (tokens_a, tokens_b)
        pop_flags = (pop_back_a, pop_back_b)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        _max = max_length - 2
        targets = (tokens_a,)
        pop_flags = (pop_back_a,)
    
    while True:
        lengths = [len(_) for _ in targets]
        if sum(lengths) <= _max:
            break
        _, t, p = sorted(zip(lengths, targets, pop_flags), key=lambda _:-_[0])[0]
        if p:
            t.pop(0)
        else:
            t.pop()
    

def write_examples_as_tfrecord(
         examples, 
         label_list, 
         max_seq_length, 
         tokenizer, 
         output_file
    ):

    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    
    def create_float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feats = from_example_to_features(
                    ex_index, 
                    example, 
                    label_list,
                    max_seq_length, 
                    tokenizer
                )

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feats.input_ids)
            features["input_mask"] = create_int_feature(feats.input_mask)
            features["segment_ids"] = create_int_feature(feats.segment_ids)
            features["label"] = create_float_feature(feats.label)
            features["is_real_example"] = create_float_feature([float(feats.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


def get_example_iterator(
        input_file, 
        seq_length, 
        num_labels,
        batch_size,
        is_training,
        drop_remainder=True,
    ):
    """
    get one-hot iterator beased on a tfrecord file
    because keras model requires fixed batch size,
    drop_remainder=True to make it decided.
    """
    
    name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label": tf.FixedLenFeature([num_labels], tf.float32),
            "is_real_example": tf.FixedLenFeature([1], tf.float32),
        }
    
    def decode_record(record):
        
        example = tf.parse_single_example(record, name_to_features)
        
        for key in example.keys():
            v = example[key]
            if v.dtype == tf.int64:
                example[key] = tf.to_int32(v)
        
        inputs = {
                'Input-Token':example['input_ids'], 
                'Input-Segment':example['segment_ids'], 
                'Input-Masked':example['input_mask']
            }
        target = example['label']
        weight = example['is_real_example'][0]
        
        return inputs, target, weight
    
    d = tf.data.TFRecordDataset(input_file)
    
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
    
    d = d.map(decode_record)
    d = d.batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder
        )
    
    return d


def padding_examples(examples, batch_size):
    while len(examples) % batch_size != 0:
        examples.append(PaddingInputExample())


def do_train(args, tokenizer, model):
    
    train_examples = args.processor.get_train_examples(args.data_dir)
    num_actual_train_examples = len(train_examples)
    padding_examples(train_examples, args.train_batch_size)
    
    num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs
        )
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    record_file = os.path.join(args.output_dir, "train.tf_record")
        
    # Augmentation on tokenization
    if args.token_sampling:
        train_examples *= int(args.num_train_epochs)
        tokenizer.enabled_sampling = True
    
    write_examples_as_tfrecord(
            train_examples,
            args.processor.get_labels(), 
            args.max_seq_length, 
            tokenizer, 
            record_file
        )
    
    if args.token_sampling:
        tokenizer.enabled_sampling = False
       
    example_iterator = get_example_iterator(
            input_file=record_file,
            seq_length=args.max_seq_length,
            num_labels=len(args.processor.get_labels()),
            batch_size=args.train_batch_size,
            is_training=True,
        )
    
    #_inputs, _target, _sample_weight = example_iterator.get_next()
    #_sample_weight.ndim = 1

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
            len(train_examples), num_actual_train_examples,
            len(train_examples) - num_actual_train_examples
        )
    tf.logging.info("  Batch size = %d", args.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    
    ckpt_path = os.path.join(
            args.output_dir,
            'weights.{epoch:02d}-{loss:.2f}.hdf5', 
        )
    callback_ckpt = keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor='loss', 
            save_best_only=False, 
            save_weights_only=False, 
            mode='auto', 
            period=args.ckpt_period,
            verbose=1, 
        )    
    
    model.fit(
            example_iterator,
            steps_per_epoch= len(train_examples) // args.train_batch_size,
            epochs=int(args.num_train_epochs),
            callbacks=[callback_ckpt],
            verbose = 1
        )


def do_eval(args, mode, tokenizer, model):
    
    if mode == 'dev':
        eval_examples = args.processor.get_dev_examples(args.data_dir)
    elif mode == 'test':
        eval_examples = args.processor.get_test_examples(args.data_dir)
    else:
        raise ValueError('unknown eval mode %s'%(mode))
    
    num_actual_eval_examples = len(eval_examples)
    
    padding_examples(eval_examples, args.eval_batch_size)
    assert len(eval_examples) % args.eval_batch_size == 0
    eval_steps = int(len(eval_examples) // args.eval_batch_size)
    
    record_file = os.path.join(args.output_dir, "%s.tf_record"%(mode))
    results_file = os.path.join(args.output_dir, "%s_results.txt"%(mode))
    
    write_examples_as_tfrecord(
            eval_examples, 
            args.processor.get_labels(), 
            args.max_seq_length, 
            tokenizer, 
            record_file
        )
    
    example_iterator = get_example_iterator(
            input_file=record_file,
            seq_length=args.max_seq_length,
            num_labels=len(args.processor.get_labels()),
            batch_size=args.eval_batch_size,
            is_training=False,
        )
    
    _inputs, _target, _sample_weight = example_iterator.get_next()
    
    # pretending to be ndarray to escape validation error at the first time
    _sample_weight.ndim=1
    
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
            len(eval_examples), num_actual_eval_examples,
            len(eval_examples) - num_actual_eval_examples
        )
    tf.logging.info("  Batch size = %d", args.eval_batch_size)
    
    if mode == 'dev':
        results = model.evaluate(
                x=_inputs,
                y=_target,
                sample_weight=[_sample_weight],
                steps=eval_steps,
                verbose = 1
            )
    
        with tf.gfile.GFile(results_file, "w") as writer:
            tf.logging.info("***** Dev results *****")
            for key, value in zip(model.metrics_names, results):
                tf.logging.info("  %s = %s", key, str(value))
                writer.write("%s = %s\n" % (key, str(value)))
    
    elif mode == 'test':
        probabilities = model.predict(
                x=_inputs,
                steps=eval_steps,
                verbose = 1
            )
        
        with tf.gfile.GFile(results_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Test results *****")
            for (i, p) in enumerate(probabilities):
                if i >= num_actual_eval_examples:
                    break
                output_line = "\t".join(
                        str(class_prob) for class_prob in p
                    ) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_eval_examples


def main(args):
    
    if not hasattr(args, 'processor'):
        args.processor = DefaultProcessor()
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    tokenization.validate_case_matches_checkpoint(
            args.do_lower_case,
            args.init_checkpoint
       )
    
    tf.gfile.MakeDirs(args.output_dir)
    
    tokenizer = tokenization.FullTokenizer(
            model_file=args.model_file, 
            vocab_file=args.vocab_file,
            do_lower_case=args.do_lower_case
       )
    
    # Specification of the custom objects used in model construction is required 
    # because the model seems to be serialized when sending TPU
    # So we use custom object scope here
    with tf.keras.utils.custom_object_scope(get_custom_objects()):
        
        model = build_model(args)
    
        if args.use_tpu:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu_grpc_url)
            strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
            model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    
        if args.do_train:
            do_train(args, tokenizer, model)
    
        if args.do_dev:
            do_eval(args, 'dev', tokenizer, model)
    
        if args.do_test:
            do_eval(args, 'test', tokenizer, model)


class Dummy(object):
    do_lower_case = True
    init_checkpoint = 'model/model.ckpt-1400000'
    output_dir = 'model/keras_debug'
    model_file = 'model/wiki-ja-mod.model'
    vocab_file = 'model/wiki-ja-mod.vocab'
    data_dir = 'data/dbdc'
    bert_config_path = 'config.json'
    use_layer_name = None
    max_seq_length=512
    train_batch_size=4
    eval_batch_size=4
    learning_rate=2e-5
    do_train=True
    do_dev=True
    do_test=True
    token_sampling=False
    warmup_proportion=0.1
    num_train_epochs=10
    dropout_rate=0.1
    use_tpu=False
    ckpt_period=2
    

if __name__ == "__main__":
    args = Dummy()
    
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session_config.gpu_options.allocator_type = "BFC"
    # session_config.gpu_options.allow_growth = True
    session = tf.Session(config=session_config)
    tf.keras.backend.set_session(session)
    main(args)

