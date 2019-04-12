# coding: utf-8


"""
Utility for livedoor corpus test
```
cd <repository root>
python src/ld_fetch_data.py
```
-h option to see detail 
"""


import sys
import os
import glob
import argparse
import configparser
import subprocess
import tarfile
import zipfile 
import tempfile
from urllib.request import urlretrieve
import json
import csv
import numpy as np
import re
import pandas as pd

# packages related to model
import tensorflow as tf
import modeling
import optimization
from utils import str_to_value
from run_dbdc_classifier import model_fn_builder
from run_dbdc_classifier import file_based_input_fn_builder
from run_dbdc_classifier import file_based_convert_examples_to_features
from run_dbdc_classifier import DefaultProcessor

import tokenization_sentencepiece
import tokenization_sp_mod

# evaluation utilities
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import math

def get_config(path=None):
    
    if path is None:
        path = 'config.ini'
    
    config = configparser.ConfigParser()
    config.read(path)
    return config


def fetch_and_decomp_raw_data(url, cache_path, ext_dir_path):
    
    if not os.path.exists(cache_path): 
        urlretrieve(url, cache_path)
    
    if not os.path.exists(ext_dir_path):
        if url.endswith('.zip'):
            with zipfile.ZipFile(cache_path) as f:
                f.extractall(ext_dir_path)
        else:
            mode = "r:gz"
            tar = tarfile.open(cache_path, mode)
            tar.extractall(ext_dir_path)
            tar.close()


def make_dataset(target_files, end_sep):
    
    def get_label(annotations):
        c = {}
        for a in annotations:
            label = a['breakdown']
            c[label] = c.get(label, 0) + 1
        sum_c = sum([_ for _ in c.values()])
        prob = {k:v/sum_c for k, v in c.items()}
        return prob
    
    def read_json(file_path):
        with open(file_path, 'r') as f:
            dialog_log = json.load(f)['turns']
        
        assert all([
                t['speaker'] == 'S' if i % 2 == 0 else 'U' 
                for i, t in enumerate(dialog_log)
            ]), 'invalid turn order found'
        if end_sep in  ''.join(
                [t['utterance'] for t in dialog_log]
            ):
            print('warn: utterances contains end_sep in %s'%(file_path))
        
        utterances = []
        srcs = []
        textas = []
        textbs = []
        probs = []
        
        for i in range(0, len(dialog_log), 2):
            if i > 0:
                utterances.append(dialog_log[i-1]['utterance'])
            srcs.append('%d@%s'%(i, file_path))
            textas.append(end_sep.join(utterances))
            textbs.append(dialog_log[i]['utterance'])
            prob = get_label(dialog_log[i]['annotations'])
            probs.append(json.dumps(prob))
            
            utterances.append(dialog_log[i]['utterance'])
        
        return list(zip(srcs, textas, textbs, probs))
    
    all_outputs = []
    for file_path in target_files:
        all_outputs.extend(read_json(file_path))
    
    return all_outputs
        

def cmd_fetch(args):
    """
    Entrypoint of fetch mode
    """
    
    # read config
    local_config = get_config()[args.config_name]
    work_dir = local_config['WORKING_DIRECTORY']
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    end_sep = json.loads(local_config['END_SEP'].strip())
    
    target_data = []
    for _, v in sorted(filter(lambda _:_[0].startswith('download_'), local_config.items())):
        v = json.loads(v)
        target_data.append((v['name'], v['url'], v['dest']))
    
    # collecting data
    collected_files = {'train':[], 'test':[]}
    for postfix, url, dest in target_data:
        d = os.path.join(work_dir, postfix.lower())
        fetch_and_decomp_raw_data(url=url, cache_path=d+'.zip', ext_dir_path=d)
        for file_path in glob.glob(d+'/**/*.json', recursive=True):
            collected_files[dest].append(file_path)
    
    # write output as csv
    header = [['src', 'text_a', 'text_b', 'prob']] 
    
    for dest in collected_files.keys():
        rows = make_dataset(collected_files[dest], end_sep)
        if dest == 'train':
            np.random.RandomState(args.random_state).shuffle(rows) 
            
            len_valid = int(len(rows)*float(local_config['VALID_PROP']))
            
            with open(os.path.join(work_dir, 'dev.csv'), 'w') as f:
                csv.writer(f).writerows(header + rows[:len_valid])
            
            with open(os.path.join(work_dir, 'train.csv'), 'w') as f:
                csv.writer(f).writerows(header + rows[len_valid:])
            
        elif dest == 'test':
            with open(os.path.join(work_dir, 'test.csv'), 'w') as f:
                csv.writer(f).writerows(header + rows)
        
        else:
            print('warn: unknown destination ignored %s'%key)
    

class Flags(object):
    """
    Make parameters to reconstruct estimator and tokenizer.
    """
    @staticmethod
    def get_latest_ckpt_path(dir_path):
        output_ckpts = glob.glob("{}/model.ckpt*.index".format(dir_path))
        latest_ckpt = sorted(
                output_ckpts, 
                key=lambda _: int(re.findall('model.ckpt-([0-9]+).index', _)[0])
            )[-1]
        return latest_ckpt.strip('.index')
    
    def __init__(self, args, config):
        # tokenizer settings
        self.model_file = os.path.join('model', args.sp_prefix+'.model')
        self.vocab_file = os.path.join('model', args.sp_prefix+'.vocab')
        self.do_lower_case = True
        
        # task processor
        self.task_proc = getattr(sys.modules[__name__], args.task_proc_name)()
        
        # model settings
        self.init_checkpoint = self.get_latest_ckpt_path(args.trained_model_path)
        self.max_seq_length = int(config['BERT-CONFIG']['max_position_embeddings'])
        self.use_tpu = False
        self.predict_batch_size = 4
        self.num_labels = len(self.task_proc.get_labels())
        
        # test dataset directory (not used for reconstruction)
        self.data_dir = args.test_data_dir
        
        # The following parameters are not used in predictions.
        # Just use to create RunConfig.
        self.output_dir = "/dummy"
        self.master = None
        self.save_checkpoints_steps = 1
        self.iterations_per_loop = 1
        self.num_tpu_cores = 1
        self.learning_rate = 0
        self.num_warmup_steps = 0
        self.num_train_steps = 0
        self.train_batch_size = 0
        self.eval_batch_size = 0


def get_from_list_to_examples(task_proc):
    """
    Return a function that converts 2d list (from csv) into example list
    This can be different between DataProcessors
    """
    if isinstance(task_proc, DefaultProcessor):
        return lambda l: task_proc._create_examples(l, "test")
    else:
        raise NotImplementedError('from_list_to_examples for %s is required '%(type(FLAGS.task_proc)))
    

def load_estimator(config, FLAGS):
    
    bert_config_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.json')
    bert_config_file.write(json.dumps({k:str_to_value(v) for k,v in config['BERT-CONFIG'].items()}))
    bert_config_file.seek(0)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file.name)
    
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host
                )
        )

    model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(FLAGS.task_proc.get_labels()),
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=FLAGS.num_train_steps,
            num_warmup_steps=FLAGS.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size
        )
    
    return estimator


def read_data(dataset_path, from_list_to_examples, class_labels, max_seq_length, use_tpu, tokenizer):
    """
    Read dataset file and prepare feature list acceptable to estimators
    """
    
    rows = []
    with open(dataset_path, 'r') as f:
        rows = [_ for _ in csv.reader(f)]
    
    input_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.tf_record')
    
    file_based_convert_examples_to_features(
            from_list_to_examples(rows),
            class_labels,
            max_seq_length, 
            tokenizer,
            input_file.name
        )
    
    input_fn = file_based_input_fn_builder(
            input_file=input_file.name,
            seq_length=max_seq_length,
            num_labels=len(class_labels),
            is_training=False,
            drop_remainder=True if use_tpu else False,
        )
    
    # Input_file object has to be kept 
    # by the caller during prediction not to be deleted
    return rows, input_file, input_fn


def evaluate(rows, label_list, prediction, by_src_label=True):
    
    report = []
   
    #
    # _majority_label, majority_label_lenient, kld, jsd, mse
    # based on https://github.com/dbd-challenge/dbdc3/blob/master/prog/eval/eval.py
    # 
    def majority_label(prob_O, prob_T, prob_X, threshold=0.0):
        if prob_O >= prob_T and prob_O >= prob_X and prob_O >= threshold:
            return "O"
        elif prob_T >= prob_O and prob_T >= prob_X and prob_T >= threshold:
            return "T"
        elif prob_X >= prob_T and prob_X >= prob_O and prob_X >= threshold:
            return "X"
        else:
            return "O"
    
    def majority_label_lenient(prob_O, prob_T, prob_X, threshold):
        if prob_O >= prob_T + prob_X and prob_O >= threshold:
            return "O_l"
        elif prob_T + prob_X >= prob_O and prob_T + prob_X >= threshold:
            return "X_l"
        else:
            return "O_l"

    def kld(p, q):
        k = 0.0
        for i in range(len(p)):
            if p[i] > 0:
                k += p[i] * (math.log(p[i]/q[i],2))

        return k

    def jsd(p, q):
        m = []
        for i in range(len(p)):
            m.append((p[i]+q[i])/2.0)

        return (kld(p,m) + kld(q,m)) / 2.0

    def mse(p, q):
        total = 0.0
        
        for i in range(len(p)):
            total += pow(p[i] - q[i],2)
    
        return total / len(p)
        
    def _append_report(df):
        report.append('Accuracy:')
        report.append(sum(df['teacher_label'] == df['pred_label']) / len(df))
        report.append('Detailed report:')
        report.append(classification_report(df['teacher_label'], df['pred_label']))
        report.append('Confusion matrix:')
        report.append(confusion_matrix(df['teacher_label'], df['pred_label']))
    
    def calc_preci_recall_f(len_p, len_g, len_pg):
        preci = len_pg / float(len_p) if len_p > 0 else 0.0
        recall = len_pg / float(len_g) if len_g > 0 else 0.0        
        f = 2*preci*recall / (preci + recall) if preci + recall > 0 else 0.0
        return preci, recall, f
    
    def _append_report2(df, threshold=0.0):
        
        num_examples = len(df)
        
        counts = {_:{'O':0, 'T':0, 'X':0, 'O_l':0, 'X_l':0} for _ in ['O', 'T', 'X']}
        
        jsd_O_T_X_sum = 0.0
        jsd_O_TX_sum = 0.0
        jsd_OT_X_sum = 0.0
        mse_O_T_X_sum = 0.0
        mse_O_TX_sum = 0.0
        mse_OT_X_sum = 0.0
        
        for i, row in df.iterrows():
                
             ans_prob_dist = (row['teacher_pO'], row['teacher_pT'], row['teacher_pX'])
             pred_prob_dist = (row['pred_pO'], row['pred_pT'], row['pred_pX'])
                
             jsd_O_T_X_sum += jsd(ans_prob_dist, pred_prob_dist)
             jsd_O_TX_sum += jsd([ans_prob_dist[0],ans_prob_dist[1] + ans_prob_dist[2]],[pred_prob_dist[0],pred_prob_dist[1] + pred_prob_dist[2]])
             jsd_OT_X_sum += jsd([ans_prob_dist[0] + ans_prob_dist[1],ans_prob_dist[2]],[pred_prob_dist[0] + pred_prob_dist[1] ,pred_prob_dist[2]])

             mse_O_T_X_sum += mse(ans_prob_dist,pred_prob_dist)
             mse_O_TX_sum += mse([ans_prob_dist[0],ans_prob_dist[1] + ans_prob_dist[2]],[pred_prob_dist[0],pred_prob_dist[1] + pred_prob_dist[2]])
             mse_OT_X_sum += mse([ans_prob_dist[0] + ans_prob_dist[1],ans_prob_dist[2]],[pred_prob_dist[0] + pred_prob_dist[1] ,pred_prob_dist[2]])
                
             pred_label = row['pred_label']   
             ans_label = majority_label(ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2], threshold)
             ans_label_l = majority_label_lenient(ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2], threshold)
             
             counts[pred_label][ans_label] += 1
             counts[pred_label][ans_label_l] += 1
        
        len_pred_X = counts['X']['O'] + counts['X']['T'] + counts['X']['X']
        len_gt_X = counts['O']['X'] + counts['T']['X'] + counts['X']['X']
        len_pred_gt_X = counts['X']['X']
        precision_s, recall_s, fmeasure_s = calc_preci_recall_f(len_pred_X, len_gt_X, len_pred_gt_X)
        report.append('Precision (X) : \t%4f (%d/%d)'%(precision_s, len_pred_gt_X, len_pred_X))
        report.append('Recall        (X) : \t%4f (%d/%d)'%(recall_s, len_pred_gt_X, len_gt_X))
        report.append('F-measure (X) : \t%4f'%(fmeasure_s))
        
        len_pred_XT = counts['X']['O_l'] + counts['X']['X_l'] + counts['T']['O_l'] + counts['T']['X_l']
        len_gt_X_l = counts['O']['X_l'] + counts['T']['X_l'] + counts['X']['X_l']
        len_pred_gt_XTX_l = counts['X']['X_l'] + counts['T']['X_l']
        precision_l, recall_l, fmeasure_l = calc_preci_recall_f(len_pred_XT, len_gt_X_l, len_pred_gt_XTX_l)
        report.append('Precision (T+X) : \t%4f (%d/%d)'%(precision_l, len_pred_gt_XTX_l,len_pred_XT))
        report.append('Recall        (T+X) : \t%4f (%d/%d)'%(recall_l, len_pred_gt_XTX_l, len_gt_X_l))
        report.append('F-measure (T+X) : \t%4f'%(fmeasure_l))

        report.append('JS divergence (O,T,X) : \t%4f'%(jsd_O_T_X_sum / num_examples))
        report.append('JS divergence (O,T+X) : \t%4f'%(jsd_O_TX_sum / num_examples))
        report.append('JS divergence (O+T,X) : \t%4f'%(jsd_OT_X_sum / num_examples))

        report.append('Mean squared error (O,T,X) : \t%4f'%(mse_O_T_X_sum / num_examples))
        report.append('Mean squared error (O,T+X) : \t%4f'%(mse_O_TX_sum / num_examples))
        report.append('Mean squared error (O+T,X) : \t%4f'%(mse_OT_X_sum / num_examples))
    
    df_src = pd.DataFrame(data=rows[1:], columns=rows[0])
    # probabilities order = O T X (defined in DefaultProcessor)
    probs = [_['probabilities'] for _ in prediction]
    df_src['pred_pO'] = [_[0] for _ in probs]
    df_src['pred_pT'] = [_[1] for _ in probs]
    df_src['pred_pX'] = [_[2] for _ in probs]
    df_src['pred_label'] = [
            majority_label(O, T, X) for O, T, X
            in zip(df_src['pred_pO'], df_src['pred_pT'], df_src['pred_pX'])
        ]
    probs = [json.loads(_) for _ in df_src['prob']]
    df_src['teacher_label'] = [majority_label(_.setdefault('O', 0), _.setdefault('T', 0), _.setdefault('X', 0)) for _ in probs]
    df_src['teacher_pO'] = [_['O'] for _ in probs]
    df_src['teacher_pT'] = [_['T'] for _ in probs]
    df_src['teacher_pX'] = [_['X'] for _ in probs]
    
    if by_src_label:
        df_src['src_label'] = [os.path.basename(os.path.dirname(_)) for _ in df_src['src']]
    
        for src_label in set(df_src['src_label']):
            report.append('***** %s *****'%(src_label))
            _append_report(df_src[df_src['src_label']==src_label])
            report.append('')
            _append_report2(df_src[df_src['src_label']==src_label])
            report.append('')
    
    else:
        _append_report(df_src)
    
    return '\n'.join([str(_) for _ in report])


def cmd_test(args):
    """
    Entrypoint of test mode
    """
    if args.trained_model_path is None:
        print('specify --trained_model_path/-p')
        sys.exit(1)
    
    if args.test_data_dir is None:
        print('specify --test_data_dir/-d')
        sys.exit(1)
    
    config = get_config()
    FLAGS = Flags(args, config)
    
    # Model
    estimator = load_estimator(config, FLAGS)
    tokenization = getattr(sys.modules[__name__], config['TOKENIZER']['PACKAGE'])
    tokenizer = tokenization.FullTokenizer(
            model_file=FLAGS.model_file,
            vocab_file=FLAGS.vocab_file,
            do_lower_case=FLAGS.do_lower_case,
        )
    
    # Test dataset
    dataset_path = os.path.join(FLAGS.data_dir, 'test.csv')
    from_list_to_examples = get_from_list_to_examples(FLAGS.task_proc)
    input_rows, input_file, input_fn = read_data(
            dataset_path=dataset_path,
            from_list_to_examples=from_list_to_examples, 
            class_labels=FLAGS.task_proc.get_labels(), 
            max_seq_length=FLAGS.max_seq_length, 
            use_tpu=FLAGS.use_tpu, 
            tokenizer=tokenizer,
        )
    
    prediction = estimator.predict(input_fn=input_fn)
    report_str = evaluate(input_rows, FLAGS.task_proc.get_labels(), prediction)
    
    print('***** RESULT *****')
    print('Test dataset:')
    print(dataset_path)
    print('Tested model:')
    print(FLAGS.init_checkpoint)
    print(report_str)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='dbdc corpus fine tuning')
    parser.add_argument('--mode','-m', choices=['fetch', 'test'], required=True, help='process mode')
    parser.add_argument('--random_state','-r', type=int, default=23, help='random state')
    parser.add_argument('--config_name','-c', type=str, default='DBDC-CORPUS', help='local config name')
    parser.add_argument('--task_proc_name','-t', type=str, default='DefaultProcessor', help='[test] Task descriptor.')
    parser.add_argument('--sp_prefix','-s', type=str, default='wiki-ja-mod', help='[test] sentencepiece model prefix')
    parser.add_argument('--trained_model_path','-p', type=str, default=None, help='[test] path to trained model directory')
    parser.add_argument('--test_data_dir','-d', type=str, default=None, help='[test] path to a directory contained test.tsv')
    args = parser.parse_args()
    
    getattr(sys.modules[__name__], 'cmd_'+args.mode)(args)

