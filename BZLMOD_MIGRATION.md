# Bzlmod Migration Summary

This document describes the migration of tdLBCpp from WORKSPACE to Bzlmod (the modern Bazel module system).

## What Changed

### 1. MODULE.bazel (Main Project)
- **Location**: `/Users/niall/Workspace/tdLBCpp/MODULE.bazel`
- **Purpose**: Declares project dependencies using Bzlmod
- **Key Features**:
  - Uses `bazel_dep` for external dependencies (rules_cc, googletest, platforms, bazel_skylib)
  - Uses `local_path_override` for local/vendored dependencies
  - Maintains backward compatibility with existing BUILD files using `repo_name`

### 2. .bazelrc Updates
- Removed WORKSPACE enable flags
- Now uses Bzlmod by default (standard for Bazel 8+)
- Comment explains that WORKSPACE is disabled

### 3. Local Module Configurations

#### rules_cuda
- **Location**: `third_party/rules_cuda/MODULE.bazel`
- **New File**: `third_party/rules_cuda/cuda/extensions.bzl`
- **Features**:
  - Created MODULE.bazel to make rules_cuda a proper Bzlmod module
  - Implemented `cuda` extension for local CUDA toolkit detection
  - Replaces WORKSPACE-based `_local_cuda` repository rule

#### tdLBGeometryRushtonTurbineLib
- **Location**: `tdLBGeometryRushtonTurbineLib/MODULE.bazel`
- **Module Name**: `tdlb_geometry_rushton_turbine_lib` (lowercase for Bzlmod compliance)
- **Repo Name**: `tdLBGeometryRushtonTurbineLib` (for BUILD file compatibility)

#### gyb
- **Location**: `third_party/gyb/MODULE.bazel`
- **Simple module declaration for the template generator**

## Benefits of Bzlmod

1. **Modern Standard**: Bzlmod is the future of Bazel (WORKSPACE deprecated in Bazel 9)
2. **Better Dependency Management**: Centralized version resolution
3. **Explicit Dependencies**: Clear declaration of all dependencies
4. **Local Path Overrides**: Easy to work with vendored/local dependencies
5. **Module Extensions**: Clean way to handle repository rules like CUDA detection

## Dependency Versions

Current versions used:
- **rules_cc**: 0.1.1
- **googletest**: 1.15.2
- **platforms**: 0.0.11
- **bazel_skylib**: 1.7.1

Local dependencies (version 0.0.0):
- **rules_cuda**: Vendored in `third_party/rules_cuda`
- **tdlb_geometry_rushton_turbine_lib**: Submodule in `tdLBGeometryRushtonTurbineLib`
- **gyb**: Vendored in `third_party/gyb`

## How It Works

### Dependency Resolution

1. **External Dependencies**: Fetched from Bazel Central Registry (BCR)
   ```starlark
   bazel_dep(name = "rules_cc", version = "0.1.1")
   ```

2. **Local Dependencies**: Overridden to point to local paths
   ```starlark
   bazel_dep(name = "rules_cuda", version = "0.0.0")
   local_path_override(
       module_name = "rules_cuda",
       path = "third_party/rules_cuda",
   )
   ```

3. **Repository Names**: Maintained for backward compatibility
   ```starlark
   bazel_dep(
       name = "tdlb_geometry_rushton_turbine_lib",
       version = "0.0.0",
       repo_name = "tdLBGeometryRushtonTurbineLib"  # BUILD files use this name
   )
   ```

### CUDA Detection

The CUDA extension (`rules_cuda/cuda/extensions.bzl`) implements a module extension that:
1. Checks for `CUDA_PATH` environment variable
2. Falls back to `which ptxas` to find CUDA installation
3. Defaults to `/usr/local/cuda` if not found
4. Creates `@local_cuda` repository for use by BUILD files

## Migration Notes

### What Still Works

- All existing BUILD files work without changes
- The Makefile continues to work
- Build scripts (`build.sh`, `test.sh`) continue to work
- All configuration profiles in `.bazelrc` still work

### WORKSPACE File

- **Status**: Still present but **not used** by Bazel 8+
- **Why Keep It**: May be useful for older Bazel versions or as reference
- **Future**: Can be deleted once fully migrated and tested

### Commands Remain the Same

```bash
# Using Make
make cpu
make gpu
make test

# Using Bazel directly
bazel build //tdlbcpp/src:tdlbcpp
bazel build --config=gpu //tdlbcpp/src:tdlbcpp
bazel test //tdlbcpp/tests/...
```

## Troubleshooting

### Version Conflicts

If you see warnings about version mismatches:
```
WARNING: For repository 'rules_cc', the root module requires module version rules_cc@0.0.13,
but got rules_cc@0.1.1 in the resolved dependency graph.
```

**Solution**: Update the version in MODULE.bazel to match the resolved version, or add `--check_direct_dependencies=off` to .bazelrc.

### Missing Repository

If you see errors like:
```
ERROR: No repository visible as '@some_repo' from repository '@@'.
```

**Solution**: Check that the dependency is declared in MODULE.bazel and, if local, has a corresponding `local_path_override`.

### Module Name Errors

If you see errors about invalid module names:
```
Error in bazel_dep: invalid module name 'MyModule'
```

**Solution**: Module names must be lowercase with only letters, digits, dots, hyphens, and underscores. Use `repo_name` parameter to maintain the original name in BUILD files.

## Testing the Migration

Build was tested and verified working:

```bash
$ bazel build //tdlbcpp/src:computeUnit
INFO: Analyzed target //tdlbcpp/src:computeUnit (71 packages loaded, 542 targets configured).
INFO: Found 1 target...
INFO: Build completed successfully, 1 total action
```

## Further Reading

- [Bazel Bzlmod Migration Guide](https://bazel.build/external/migration)
- [MODULE.bazel File Format](https://bazel.build/rules/lib/globals/module)
- [Module Extensions](https://bazel.build/rules/lib/globals/module#module_extension)
- [Local Path Override](https://bazel.build/rules/lib/globals/module#local_path_override)
