# Code Formatting Guide

This guide explains how to install and use clang-format for the tdLBCpp project.

## Installation

### macOS

```bash
# Install clang-format via Homebrew
brew install clang-format

# Verify installation
clang-format --version
```

### Linux (Ubuntu/Debian)

```bash
# Install clang-format
sudo apt-get update
sudo apt-get install clang-format

# Or install a specific version (e.g., 15)
sudo apt-get install clang-format-15

# Verify installation
clang-format --version
```

### Linux (Fedora/RHEL)

```bash
# Install clang-format
sudo dnf install clang-tools-extra

# Verify installation
clang-format --version
```

## Usage

### Using the Makefile (Recommended)

```bash
# Check if code is properly formatted
make format-check

# Automatically fix formatting issues
make format

# Format a specific file or directory
make format-path PATH_TO_FORMAT=tdlbcpp/src/Params/FlowParams.hpp
```

### Using the Script Directly

```bash
# Check formatting (returns error if files need formatting)
./format-code.sh --check

# Fix formatting issues
./format-code.sh --fix

# Check/fix specific directory
./format-code.sh --check tdlbcpp/src/Params
./format-code.sh --fix tdlbcpp/src/Params
```

### Using clang-format Directly

```bash
# Check a single file (dry-run)
clang-format --dry-run --Werror tdlbcpp/src/Params/FlowParams.hpp

# Format a single file (in-place)
clang-format -i tdlbcpp/src/Params/FlowParams.hpp

# Show diff of what would change
clang-format --dry-run tdlbcpp/src/Params/FlowParams.hpp | diff - tdlbcpp/src/Params/FlowParams.hpp
```

## IDE Integration

### Visual Studio Code

#### Setup
1. Install the **C/C++** extension by Microsoft (ms-vscode.cpptools)
2. Create/update `.vscode/settings.json` in the project root:

```json
{
    "C_Cpp.clang_format_path": "/opt/homebrew/bin/clang-format",
    "C_Cpp.clang_format_style": "file",
    "C_Cpp.clang_format_fallbackStyle": "Google",
    "editor.formatOnSave": true,
    "editor.formatOnType": false,
    "[cpp]": {
        "editor.defaultFormatter": "ms-vscode.cpptools",
        "editor.tabSize": 4
    },
    "[c]": {
        "editor.defaultFormatter": "ms-vscode.cpptools",
        "editor.tabSize": 4
    },
    "files.associations": {
        "*.hpp": "cpp",
        "*.cuh": "cuda-cpp"
    }
}
```

**Note**: Update `clang_format_path` to match your installation:
- macOS (Intel): `/usr/local/bin/clang-format`
- macOS (ARM): `/opt/homebrew/bin/clang-format`
- Linux: `/usr/bin/clang-format`

#### Usage
- **Format entire file**: `Shift+Alt+F` (or `Shift+Option+F` on macOS)
- **Format selection**: Select code, then `Shift+Alt+F`
- **Auto-format on save**: Enabled in settings above

#### Optional: Format on Paste
Add to settings.json:
```json
{
    "editor.formatOnPaste": true
}
```

### CLion / IntelliJ IDEA

#### Setup
1. Go to: **Preferences → Editor → Code Style → C/C++**
2. Click **"Set from..."** → **"ClangFormat"**
3. Check **"Enable ClangFormat (only for C/C++/Objective-C)"**
4. Set **"clangd format style"** to **"File"**

Or use the external formatter:
1. Go to: **Preferences → Tools → External Tools**
2. Click **+** to add a new tool
3. Configure:
   - **Name**: Format with clang-format
   - **Program**: `clang-format`
   - **Arguments**: `-i $FilePath$`
   - **Working directory**: `$ProjectFileDir$`

#### Usage
- **Format file**: `Cmd+Alt+L` (macOS) or `Ctrl+Alt+L` (Linux/Windows)
- **Format selection**: Select code, then use format shortcut
- **External tool**: Right-click → External Tools → Format with clang-format

### Xcode

#### Setup
1. Install **ClangFormat-Xcode** plugin:
   ```bash
   brew install --cask clangformat-xcode
   ```
2. Restart Xcode
3. Go to: **Editor → ClangFormat → File**

#### Usage
- **Format file**: Select **Editor → ClangFormat → Format File**
- **Format selection**: Select code, then **Editor → ClangFormat → Format Selection**

### Vim/Neovim

#### Setup with vim-autoformat
Add to your `.vimrc` or `init.vim`:

```vim
" Install vim-autoformat plugin first
" Using vim-plug:
Plug 'Chiel92/vim-autoformat'

" Configure clang-format
let g:formatdef_clangformat = '"clang-format -style=file"'
let g:formatters_cpp = ['clangformat']
let g:formatters_c = ['clangformat']

" Format on save (optional)
au BufWrite *.cpp,*.hpp,*.h,*.cuh :Autoformat

" Manual format command
nnoremap <leader>f :Autoformat<CR>
vnoremap <leader>f :Autoformat<CR>
```

#### Setup without plugins (simple)
```vim
" Format entire file
nnoremap <leader>f :%!clang-format<CR>

" Format selection
vnoremap <leader>f :!clang-format<CR>

" Format on save (optional)
autocmd BufWritePre *.cpp,*.hpp,*.h,*.cuh :silent! %!clang-format
```

#### Neovim with LSP (modern setup)
Add to your `init.lua`:

```lua
-- Configure clangd LSP with formatting
require('lspconfig').clangd.setup({
    cmd = {
        "clangd",
        "--clang-tidy",
        "--completion-style=detailed",
    },
    on_attach = function(client, bufnr)
        -- Enable formatting
        client.server_capabilities.documentFormattingProvider = true

        -- Format on save
        vim.api.nvim_create_autocmd("BufWritePre", {
            buffer = bufnr,
            callback = function()
                vim.lsp.buf.format({ async = false })
            end,
        })

        -- Manual format keybinding
        vim.keymap.set('n', '<leader>f', vim.lsp.buf.format, { buffer = bufnr })
    end,
})
```

### Emacs

#### Setup with clang-format.el
Add to your `.emacs` or `init.el`:

```elisp
;; Install clang-format.el package first
(require 'clang-format)

;; Use system clang-format binary
(setq clang-format-executable "clang-format")

;; Keybindings
(global-set-key [C-M-tab] 'clang-format-region)
(global-set-key [C-S-tab] 'clang-format-buffer)

;; Format on save (optional)
(add-hook 'c++-mode-hook
    (lambda ()
        (add-hook 'before-save-hook 'clang-format-buffer nil 'local)))
(add-hook 'c-mode-hook
    (lambda ()
        (add-hook 'before-save-hook 'clang-format-buffer nil 'local)))
```

#### Using format-all-mode (alternative)
```elisp
(require 'format-all)
(add-hook 'c++-mode-hook #'format-all-mode)
(add-hook 'c-mode-hook #'format-all-mode)
```

### Sublime Text

#### Setup
1. Install **Package Control** if not already installed
2. Install **Clang Format** package:
   - `Cmd+Shift+P` → "Package Control: Install Package" → "Clang Format"
3. Configure: **Preferences → Package Settings → Clang Format → Settings - User**:

```json
{
    "binary": "/opt/homebrew/bin/clang-format",
    "style": "file",
    "format_on_save": true
}
```

#### Usage
- **Format file**: `Cmd+Alt+A` (macOS) or `Ctrl+Alt+A` (Linux/Windows)

### Eclipse CDT

#### Setup
1. Install **CppStyle** plugin:
   - Help → Eclipse Marketplace → Search "CppStyle" → Install
2. Configure: **Window → Preferences → C/C++ → CppStyle**
   - **Clang-format path**: `/opt/homebrew/bin/clang-format`
   - **Enable**: Format on save
   - **Style**: file

#### Usage
- **Format file**: `Cmd+Shift+F` (macOS) or `Ctrl+Shift+F` (Linux/Windows)

### Kate / KDevelop

#### Setup
1. Go to: **Settings → Configure Kate → Plugins**
2. Enable **"Format"** plugin
3. Go to: **Settings → Configure Kate → Format**
4. Add **clang-format** formatter:
   - **Command**: `clang-format`
   - **Arguments**: `-style=file`

#### Usage
- **Format**: Tools → Format

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/format-check.yml`:

```yaml
name: Format Check
on: [push, pull_request]

jobs:
  format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install clang-format
        run: sudo apt-get install -y clang-format
      - name: Check formatting
        run: |
          ./format-code.sh --check
          if [ $? -ne 0 ]; then
            echo "❌ Format check failed. Run 'make format' locally."
            exit 1
          fi
```

### GitLab CI

Add to `.gitlab-ci.yml`:

```yaml
format-check:
  stage: test
  image: ubuntu:latest
  before_script:
    - apt-get update && apt-get install -y clang-format
  script:
    - ./format-code.sh --check
  allow_failure: false
```

## Configuration

The project uses `.clang-format` for configuration. Key settings:

- **Indentation**: 4 spaces
- **Line length**: 100 characters
- **Brace style**: Attach (K&R style)
- **Pointer alignment**: Left (`int* ptr`)
- **Include sorting**: Enabled with categories

To customize, edit `.clang-format` in the project root.

## Troubleshooting

### Command not found

If you get `command not found` after installation:

```bash
# macOS: Check Homebrew installation
which clang-format

# If not found, add Homebrew to PATH
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Wrong version

Different clang-format versions may format code differently:

```bash
# Check version
clang-format --version

# The project is tested with clang-format 15+
# Install specific version if needed:
brew install llvm@15
```

### Files not being formatted

Check that:
1. The file extensions match: `.cpp`, `.hpp`, `.h`, `.cuh`
2. Files aren't in excluded directories: `bazel-*`, `build`, `third_party`
3. The `.clang-format` file exists in the project root

### Format breaks compilation

If formatting causes issues:
1. Check compiler warnings - the code may have been incorrect
2. Verify `.clang-format` settings match your C++ standard (C++17)
3. File a bug report with the specific file and issue

## Best Practices

1. **Format before committing**: Always run `make format-check` before committing
2. **Format incrementally**: Format files as you edit them, not all at once
3. **Review formatting changes**: Don't blindly accept all formatting changes
4. **Keep formatting separate**: Make formatting changes in separate commits
5. **Team agreement**: Ensure team agrees on `.clang-format` settings

## Example Workflow

```bash
# Edit some files
vim tdlbcpp/src/Params/FlowParams.hpp

# Check formatting
make format-check

# Fix formatting if needed
make format

# Review changes
git diff

# Commit
git add tdlbcpp/src/Params/FlowParams.hpp
git commit -m "Fix formatting in FlowParams.hpp"

# Or commit formatting separately
git add -p  # Select only non-formatting changes
git commit -m "Add new feature"

# Then format in separate commit
make format
git add -u
git commit -m "Apply clang-format"
```

## Resources

- [clang-format documentation](https://clang.llvm.org/docs/ClangFormat.html)
- [.clang-format options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [clang-format configurator](https://zed0.co.uk/clang-format-configurator/)
