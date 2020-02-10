#!/bin/bash

stage=-1
egs_dir=egs/feats

if [ $# != 1 ];then
  echo "usage: %0 stage"
  exit 1
fi

stage=$1

if [[ $stage -le -1 && $stage -ge -1 ]]; then
  echo "generate fake kaldi spaker feats..."
  rm -rf $egs_dir/data
  mkdir -p $egs_dir/data
  $egs_dir/local/make_spk_feats.py  $egs_dir
  $egs_dir/utils/utt2spk_to_spk2utt.pl $egs_dir/data/utt2spk > $egs_dir/data/spk2utt
fi

if [ $stage -le 0 ] && [ $stage -ge 0 ];then
  echo "build create_speaker_features..."
  bazel build lingvo/tools:create_speaker_features || exit 1
  echo "dump spk2id map..."
  ./bazel-bin/lingvo/tools/create_speaker_features \
    --logtostderr \
    --dump_spk2id \
    --input_spk2utt $egs_dir/data/spk2utt \
    --spk2id_filepath $egs_dir/data/spk2id
  echo "dump feats to tfrecords..."
  ./bazel-bin/lingvo/tools/create_speaker_features \
    --logtostderr \
    --generate_tfrecords \
    --input_spk2utt $egs_dir/data/spk2utt \
    --input_feats $egs_dir/data/feats.scp \
    --shard_id=0 --num_shards=1  \
    --num_output_shards=1 --output_range_begin=0 --output_range_end=1 \
    --output_template="$egs_dir/data/feats.tfrecords-%5.5d-of-%5.5d"

fi

if [ $stage -le 1 ] && [ $stage -ge 1 ];then
  bazel build //lingvo/tools:print_tf_records
  echo "print tfrecords..."
  bazel-bin/lingvo/tools/print_tf_records --input_filepattern "$egs_dir/data/feats.tfrecords-00000-of-00001"
fi

