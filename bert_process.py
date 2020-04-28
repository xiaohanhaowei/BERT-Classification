# coding = utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import modeling
import optimization
import tokenization
import tensorflow as tf
import sys
import os
import bert
from config import FLAGS
from classifier_utils import *
datas = '韩月华报：在牡丹园地铁站东南口电动车被盗;经与报警人核实称，5月25日16时许将电动车停放在牡丹园地铁站东南口，现发现车子不见了，求助民警。值班领导：刘超，回复：许志伟。'


def datas_to_bert(datas, mode_type):
  '''
  @Author: hongwei.wang
  @date: 2020.04.16
  @func:
    process the datas and directly input the content of the datas into the bert model
  args:

  returns:
  '''
  datas_for_bert = bert.data_prepare.local_based_dataset_prepare(datas)
  return datas_for_bert


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
  num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def BERT_Inference(local_data_for_bert, do_predict=True):
  '''
  @Author: hongwei.wang
  @date: 2020.04.16
  @func:
    BERT's inference
  args:

  returns: the softmax output
  '''
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "iflytek": iFLYTEKDataProcessor,
      "iflytek_deriv": iFLYTEKDERIVDataProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

  if not do_predict:
    raise ValueError(
        "'do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  # added by hongwei.wang 
  # TODO: need to modify the label_list
  label_list = processor.get_labels(FLAGS.data_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  # if FLAGS.use_tpu and FLAGS.tpu_name:
  #   tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
  #       FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  # train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  #   print("data_dir:", FLAGS.data_dir)
  #   train_examples = processor.get_train_examples(FLAGS.data_dir)
  #   num_train_steps = int(
  #       len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  #   num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples_from_local(local_data_for_bert)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    if task_name == "inews":
      file_based_convert_examples_to_features_for_inews(predict_examples, label_list,
                                                        FLAGS.max_seq_length, tokenizer,
                                                        predict_file)
    else:
      file_based_convert_examples_to_features(predict_examples, label_list,
                                              FLAGS.max_seq_length, tokenizer,
                                              predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    index2label_map = {}
    for (i, label) in enumerate(label_list):
      index2label_map[i] = label
    output_predict_file_label_name = task_name + "_predict.json"
    output_predict_file_label = os.path.join(FLAGS.output_dir, output_predict_file_label_name)
    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file_label, "w") as writer_label:
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(result):
          probabilities = prediction["probabilities"]
          label_index = probabilities.argmax(0)
          if i >= num_actual_predict_examples:
            break
          output_line = "\t".join(
              str(class_probability)
              for class_probability in probabilities) + "\n"
          test_label_dict = {}
          test_label_dict["id"] = i
          test_label_dict["label"] = str(index2label_map[label_index])
          if task_name == "tnews":
            test_label_dict["label_desc"] = ""
          writer.write(output_line)
          json.dump(test_label_dict, writer_label)
          writer_label.write("\n")
          num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples
