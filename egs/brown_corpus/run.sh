
stage=$1
model=punctuator.codelab.RNMTModel
#logdir=$PWD/egs/brown_corpus/exp/$model
logdir=/nfs/project/zhanghui/lingvo/brown_corpus/logdir
mkdir -p logdir

echo stage: $stage
echo workdir: $PWD
echo model: $model
echo logdir: $logdir

if [ $stage -le 0 ] && [ $stage -ge 0 ];then
  # gen data 
  bazel run -c opt --distdir=~/dist lingvo/tasks/punctuator/tools:download_brown_corpus --  --outdir="/nfs/project/datasets/opensource_data/brown_corpus"
fi

if [ $stage -le 1 ] && [ $stage -ge 1 ];then
   # train and eval
   echo "Training..."
   bazel run -c opt --distdir=~/dist --config=cuda //lingvo:trainer --  \
    --mode=sync    \
    --job=controller,trainer_client \
    --run_locally=gpu    \
    --worker_gpus=1   \
    --worker_split_size=1  \
    --model=$model \
    --logdir=$logdir \
    --saver_max_to_keep=10  \
    --saver_keep_checkpoint_every_n_hours=0.0 \
    --alsologtostderr
fi

if [ $stage -le 2 ] && [ $stage -ge 2 ];then
  echo "Eval test..."
  bazel run -c opt --distdir=~/dist --config=cuda //lingvo:trainer --  \
    --mode=sync    \
    --job=evaler_test\
    --run_locally=cpu     \
    --model=$model \
    --logdir=$logdir \
    --alsologtostderr
fi

if [ $stage -le 3 ] && [ $stage -ge 3 ];then
  echo "Decode test..."
  bazel run -c opt --distdir=~/dist --config=cuda //lingvo:trainer --  \
    --mode=sync    \
    --job=decoder_test \
    --run_locally=cpu     \
    --model=$model \
    --logdir=$logdir \
    --alsologtostderr
fi


if [ $stage -le 4 ] && [ $stage -ge 4 ];then
  # dump inference graph
  echo "Dump Model..."
  bazel run -c opt --distdir=~/dist --config=cuda //lingvo:trainer --  \
    --mode=write_inference_graph  \
    --run_locally=cpu     \
    --model=$model \
    --logdir=$logdir \
    --alsologtostderr
fi

if [ $stage -le 5 ] && [ $stage -ge 5 ];then
  # interactive predictor
  CUDA_VISIBLE_DEVICES= bazel run lingvo/core:predictor

  #pred = Predictor(inference_graph=inference_graph)
  #pred.Load("/tmp/logdir/train/ckpt-00000000")
  #[topk_hyps] = pred.Run(["topk_hyps"], src_strings=["Hello World"])
fi

if [ $stage -le 6 ] && [ $stage -ge 6 ];then
  echo "Test..."
  CUDA_VISIBLE_DEVICES= bazel test --test_output=all lingvo/tasks/punctuator/...
fi