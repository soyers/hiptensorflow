# hipTensorflow Quickstart Guide

In this quickstart guide, we'll walk through the steps for ROCm installation, run a few Tensorflow workloads, and discuss FAQs and tips.  

## Install ROCm & Tensorflow

For basic installation instructions for ROCm and Tensorflow, please see [this doc](tensorflow-install-basic.md).

## Workloads

### Soumith Convnet Benchmarks
Details on the convnet benchmarks can be found at this [link](https://github.com/soumith/convnet-benchmarks).

Here are the basic instructions:  
```
# Grab the code
cd $HOME 
git clone https://github.com/soumith/convnet-benchmarks.git
BENCHDIR=$HOME/convnet-benchmarks/tensorflow
cd hiptensorflow

# Run the benchmarks
rm -f $BENCHDIR/output_*.log
MODELS="alexnet overfeat vgg googlenet"
for m in $MODELS
do
    python $BENCHDIR/benchmark_${m}.py 2>&1 | tee $BENCHDIR/output_${m}.log
done

# Get a quick summary
find $BENCHDIR -name "output*.log" -print -exec grep 'across 100 steps' {} \; -exec echo \;
```

Expected result:  
```
/root//convnet-benchmarks/tensorflow/output_alexnet.log
2017-09-16 20:21:03.171504: Forward across 100 steps, 0.025 +/- 0.003 sec / batch
2017-09-16 20:21:17.314033: Forward-backward across 100 steps, 0.084 +/- 0.008 sec / batch

/root//convnet-benchmarks/tensorflow/output_googlenet.log
2017-09-16 20:24:49.543523: Forward across 100 steps, 0.100 +/- 0.010 sec / batch
2017-09-16 20:27:46.143870: Forward-backward across 100 steps, 0.328 +/- 0.033 sec / batch

/root//convnet-benchmarks/tensorflow/output_overfeat.log
2017-09-16 20:21:36.272601: Forward across 100 steps, 0.096 +/- 0.010 sec / batch
2017-09-16 20:22:18.039344: Forward-backward across 100 steps, 0.340 +/- 0.034 sec / batch

/root//convnet-benchmarks/tensorflow/output_vgg.log
2017-09-16 20:22:40.406108: Forward across 100 steps, 0.132 +/- 0.001 sec / batch
2017-09-16 20:23:30.809168: Forward-backward across 100 steps, 0.384 +/- 0.002 sec / batch
```

## FAQs & tips

### Temp workaround:  Solutions when running out of memory
As a temporary workaround, if your workload runs out of device memory, you can either reduce the batch size or set `config.gpu_options.allow_growth = True`.