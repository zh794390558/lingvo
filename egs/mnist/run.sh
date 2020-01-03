
stage=$1
model=image.mnist.SmoothLeNet5
logdir=$PWD/egs/mnist/exp/$model

echo stage: $stage
echo workdir: $PWD
echo model: $model
echo logdir: $logdir

if [ $stage -le 0 ] && [ $stage -ge 0 ];then
  # gen data 
  bazel run -c opt --distdir=~/dist //lingvo/tools:keras2ckpt --  --dataset=mnist
fi

if [ $stage -le 1 ] && [ $stage -ge 1 ];then
  # train and eval
  bazel run -c opt --distdir=~/dist --config=cuda //lingvo:trainer --  \
    --mode=sync    \
    --job=controller,trainer_client,evaler_dev,decoder_test \
    --run_locally=cpu     \
    --saver_max_to_keep=10     \
    --saver_keep_checkpoint_every_n_hours=100000.0    \
    --worker_gpus=0   \
    --worker_split_size=1     \
    --model=$model \
    --logdir=$logdir \
    --alsologtostderr
fi

if [ $stage -le 2 ] && [ $stage -ge 2 ];then
  # dump inference graph
  bazel run -c opt --distdir=~/dist --config=cuda //lingvo:trainer --  \
    --mode=write_inference_graph  \
    --run_locally=cpu     \
    --saver_max_to_keep=10     \
    --saver_keep_checkpoint_every_n_hours=100000.0    \
    --worker_gpus=0   \
    --worker_split_size=1     \
    --model=$model \
    --logdir=$logdir \
    --alsologtostderr
fi

if [ $stage -le 3 ] && [ $stage -ge 3 ];then
  # interactive predictor
  CUDA_VISIBLE_DEVICES= bazel run lingvo/core:predictor

  #pred = Predictor(inference_graph=inference_graph)
  #pred.Load("/tmp/logdir/train/ckpt-00000000")
  #[topk_hyps] = pred.Run(["topk_hyps"], src_strings=["Hello World"])
fi
