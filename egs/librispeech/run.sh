bazel run -c opt --distdir=~/dist --config=cuda //lingvo:trainer --  \
  --alsologtostderr  \
  --mode=shell    \
  --run_locally=gpu     \
  --saver_max_to_keep=10     \
  --saver_keep_checkpoint_every_n_hours=100000.0    \
  --worker_gpus=1     \
  --worker_split_size=1     \
  --model=asr.librispeech.Librispeech960Grapheme \
  --logdir=./ckpt/libri/grapheme
