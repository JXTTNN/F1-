#!/usr/bin/env bash
# 把本次 CI 的诊断结论写进 ``ci-status`` 分支。
#
# 动机：本机到 GitHub API 常常不通/被限流，判断「云端到底哪个用例红了」
# 只能靠 git。与其等人在网页上翻日志，不如让云端自己把结论提交回仓库：
#
#     git fetch origin ci-status:refs/remotes/origin/ci-status
#     git show origin/ci-status:ci-status/CI__test.txt
#
# 关键点：多个工作流会同时往这个分支写（CI / Full CI / Retrain / Build），
# 所以**不能**各自从自己的 HEAD 拉一条新分支再强推 —— 那样后写的会把先写的
# 整个冲掉（本脚本第一版就踩了这个坑：Full CI 的上报把 CI 的上报覆盖没了）。
# 这里改用底层命令：以远端 ci-status 的树为基线，只覆盖本 job 那个文件，
# 再 commit-tree 成新提交后强推。谁最后写都不影响别人的结论。
#
# 写入失败绝不影响主流程（诊断是附加能力，不能反过来把构建搞红）。
#
# 用法：bash scripts/ci_diag.sh <workflow 名> <job 名> [步骤结论]
#
# 第三个参数通常由工作流传入各步骤的 outcome（success/failure/skipped），
# 例如 `lint=success js=failure pytest=skipped`。没有它时「pytest 之前就失败了」
# 只能靠猜是哪一步（实测踩过：ruff / node --check / pip install 都可能是凶手）。
set -u

WF="${1:-ci}"
JOB="${2:-job}"
STEPS="${3:-}"
SAFE_WF=$(printf '%s' "$WF" | tr ' /&' '___')
SAFE_JOB=$(printf '%s' "$JOB" | tr ' /&' '___')
DIR="ci-status"
OUT="$DIR/${SAFE_WF}__${SAFE_JOB}.txt"

mkdir -p "$DIR"

{
  echo "workflow : $WF"
  echo "job      : $JOB"
  echo "sha      : ${GITHUB_SHA:-unknown}"
  echo "ref      : ${GITHUB_REF_NAME:-unknown}"
  echo "event    : ${GITHUB_EVENT_NAME:-unknown}"
  echo "run      : ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-unknown}/actions/runs/${GITHUB_RUN_ID:-0}"
  echo "time     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ -n "$STEPS" ]; then
    echo "steps    : $STEPS"
  fi
  echo

  shopt -s nullglob
  logs=(.ci/*.log)
  if [ ${#logs[@]} -eq 0 ]; then
    echo "(本 job 没有产出 .ci/*.log —— 失败发生在 pytest 之前的步骤："
    echo " ${STEPS:-未提供步骤结论})"
  fi

  for f in "${logs[@]}"; do
    echo "================ $f ================"
    echo "---- 失败 / 错误用例 ----"
    grep -E "^(FAILED|ERROR) " "$f" | head -40 || true
    echo "---- 断言摘要 ----"
    grep -E "^E  " "$f" | head -40 || true
    echo "---- 最慢用例 ----"
    awk '/slowest .*durations/{p=1} p' "$f" | head -25 || true
    echo "---- 结果行 ----"
    grep -E "[0-9]+ (passed|failed)" "$f" | tail -3 || true
    echo "---- 超时 / 崩溃痕迹 ----"
    grep -E "Timeout|timed out|SIGKILL|Killed|MemoryError" "$f" | head -10 || true
    echo "---- 日志尾部 ----"
    tail -60 "$f" || true
    echo
  done
} > "$OUT"

git config user.name "ci-status-bot"
git config user.email "ci-status-bot@users.noreply.github.com"

# 取远端 ci-status 的当前顶端做基线（不存在就是首次写入）。
BASE=""
if git fetch -q origin ci-status:refs/remotes/origin/ci-status 2>/dev/null; then
  BASE=$(git rev-parse --verify -q refs/remotes/origin/ci-status || true)
fi

if [ -n "$BASE" ]; then
  git read-tree "$BASE" >/dev/null 2>&1 || exit 0
else
  git read-tree --empty >/dev/null 2>&1 || exit 0
fi

# 只更新本 job 这一个路径，**不要**用 `git add ci-status`：base 树里别人的
# 报告文件在工作区并不存在，`git add <dir>` 会把这些「缺失」当成删除写进
# 索引，等于又把别人的结论抹掉了（这是本脚本第二次踩的坑）。
BLOB=$(git hash-object -w "$OUT" 2>/dev/null) || exit 0
git update-index --add --cacheinfo "100644,${BLOB},${OUT}" >/dev/null 2>&1 || exit 0
TREE=$(git write-tree 2>/dev/null) || exit 0

MSG="ci-status: ${WF}/${JOB} @ ${GITHUB_SHA:0:7}"
if [ -n "$BASE" ]; then
  COMMIT=$(git commit-tree "$TREE" -p "$BASE" -m "$MSG" 2>/dev/null) || exit 0
else
  COMMIT=$(git commit-tree "$TREE" -m "$MSG" 2>/dev/null) || exit 0
fi

for _ in 1 2 3; do
  if git push -f origin "$COMMIT:refs/heads/ci-status" >/dev/null 2>&1; then
    echo "诊断已写入 ci-status 分支：${OUT}"
    exit 0
  fi
  sleep 5
done

echo "::warning title=诊断写入失败::ci-status 分支推送失败（不影响构建结论）"
exit 0
