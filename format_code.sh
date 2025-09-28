#!/bin/bash

# --- 配置项 ---
# Clang-Format 可执行文件的路径。
# 如果你已经将其添加到系统PATH，直接使用 "clang-format" 即可。
# 否则，请提供完整的路径，例如："/usr/bin/clang-format-15" 或 "/opt/homebrew/bin/clang-format"。
CLANG_FORMAT_BIN="/usr/bin/clang-format-15"

# 要格式化的文件扩展名列表。
# 可以根据你的项目需求添加或删除。
# 例如：'cpp\|hpp\|cc\|cxx\|h\|c\|cu\|cuh'
FILE_EXTENSIONS='cpp\|hpp\|cc\|cxx\|h\|c\|cu\|cuh'

# 排除的目录列表，用 | 分隔（用于正则表达式）。
# 这个例子排除了 '3rdparty' 目录。如果你有其他需要排除的目录，可以添加到这里。
EXCLUDE_DIR1='3rdparty'
EXCLUDE_DIR2='build'

# --- 脚本逻辑 ---

echo "--- 开始使用 clang-format 格式化项目文件 ---"
echo "正在查找文件类型: .$FILE_EXTENSIONS"
echo "将排除的目录: $EXCLUDE_DIRS"
echo "将使用的 clang-format 工具: $(which $CLANG_FORMAT_BIN 2>/dev/null || echo '未找到或不在PATH中')"

# 检查 clang-format 是否可用
if ! command -v "$CLANG_FORMAT_BIN" &> /dev/null
then
    echo "错误: 未找到 '$CLANG_FORMAT_BIN'。请检查 CLANG_FORMAT_BIN 变量或确保 clang-format 已安装并位于 PATH 中。"
    exit 1
fi

# 查找并格式化文件
# -type f 查找文件
# -regex ".*\\.\($FILE_EXTENSIONS\)" 匹配指定扩展名
# -not -path "./$EXCLUDE_DIRS/*" 排除指定目录下的文件
# -print0 使用空字符分隔文件名，以便处理带空格的文件名
find . -type f \
    -regex ".*\\.\($FILE_EXTENSIONS\)" \
    -not -path "./$EXCLUDE_DIR1/*" \
    -not -path "./$EXCLUDE_DIR2/*" \
    -print0 | while IFS= read -r -d $'\0' file; do
    echo "格式化: $file"
    "$CLANG_FORMAT_BIN" -style=file -i "$file"
done

echo "--- 格式化完成 ---"