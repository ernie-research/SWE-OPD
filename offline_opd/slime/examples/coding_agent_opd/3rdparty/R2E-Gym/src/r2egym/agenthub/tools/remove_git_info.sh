#!/bin/bash

# 默认值
REMOVE_TAG=true

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --remove_tag)
            if [ "$2" = "true" ] || [ "$2" = "false" ]; then
                REMOVE_TAG=$2
                shift 2
            else
                echo "错误: --remove_tag 参数必须是 true 或 false"
                exit 1
            fi
            ;;
        *)
            TARGET_COMMIT=$1
            shift
            ;;
    esac
done

# 检查是否输入了 commit_id
if [ -z "$TARGET_COMMIT" ]; then
    echo "使用方法: $0 <commit_id> [--remove_tag true|false]"
    echo "  --remove_tag: 是否删除标签 (默认: true)"
    exit 1
fi

echo "--- 1. 正在保护本地修改 (Stashing) ---"
# --include-untracked 会连同未跟踪的文件一起存入进度
# 如果没有修改，stash 会返回失败，所以加个判断
STASH_RESULT=$(git stash push --include-untracked -m "Pre-sanitization backup")

echo "--- 2. 开始清理未来信息 ---"
# 强制重置到目标提交
# git reset --hard $TARGET_COMMIT

# 删除远程、其他分支
git remote remove origin 2>/dev/null
git branch | grep -v "^\*" | xargs -r git branch -D

# 根据参数决定是否删除标签
if [ "$REMOVE_TAG" = "true" ]; then
    echo "正在删除标签..."
    git tag | xargs -r git tag -d
else
    echo "跳过删除标签 (--remove_tag=false)"
fi

# 彻底清理引用日志和对象，防止泄露
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "--- 3. 恢复本地修改 (Unstashing) ---"
# 如果之前成功 stash 了，现在恢复它
if [[ "$STASH_RESULT" != "No local changes to save" ]]; then
    git stash pop
    echo "本地修改已还原。"
else
    echo "之前没有检测到需要保存的修改。"
fi

echo "--- 清理完成！已锁定至 $TARGET_COMMIT ---"