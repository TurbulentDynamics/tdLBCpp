"""Bzlmod extensions for rules_cuda."""

load("//cuda:dependencies.bzl", "rules_cuda_dependencies")

def _local_cuda_impl(repository_ctx):
    """Implementation of local_cuda repository rule."""
    # Path to CUDA Toolkit is
    # - taken from CUDA_PATH environment variable or
    # - determined through 'which ptxas' or
    # - defaults to '/usr/local/cuda'
    cuda_path = "/usr/local/cuda"
    ptxas_path = repository_ctx.which("ptxas")
    if ptxas_path:
        cuda_path = ptxas_path.dirname.dirname
    cuda_path = repository_ctx.os.environ.get("CUDA_PATH", cuda_path)

    defs_template = "def if_local_cuda(true, false = []):\n    return %s"
    if repository_ctx.path(cuda_path).exists:
        repository_ctx.symlink(cuda_path, "cuda")
        repository_ctx.symlink(Label("//private:BUILD.local_cuda"), "BUILD")
        repository_ctx.file("defs.bzl", defs_template % "true")
    else:
        repository_ctx.file("BUILD")  # Empty file
        repository_ctx.file("defs.bzl", defs_template % "false")

_local_cuda = repository_rule(
    implementation = _local_cuda_impl,
    environ = ["CUDA_PATH", "PATH"],
)

def _cuda_extension_impl(module_ctx):
    """Module extension for CUDA dependencies."""
    _local_cuda(name = "local_cuda")

cuda = module_extension(
    implementation = _cuda_extension_impl,
)
