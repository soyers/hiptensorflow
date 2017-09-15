# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with tensorflow ops.
#   APIs are meant to change over time.
package(
    default_visibility = ["//visibility:private"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_cc_test",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

tf_custom_op_library(
    name = "python/ops/_nccl_ops.so",
    srcs = [
    ],
    deps = [
    ],
)
